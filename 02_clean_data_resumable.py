"""
Stage 3 — Build Targets and Features (memory-safe, drive-partitioned, resumable)

WHY THIS IS DIFFERENT FROM A "LOAD THE FULL DATASET" VERSION:
The straightforward version of this stage combines all cleaned quarterly
parquet files into one big frame before computing anything. On this
dataset (~682M rows, ~110 SMART columns after Stage 2's cleaning), that's
roughly 300GB of RAM just to hold the raw data once -- completely
infeasible on a 15.2GB machine, and far worse than any single-quarter OOM
hit during Stage 2.

The fix follows the same principle Stage 2 ended up needing: every
computation here (failure date, RUL, rolling features, tenure features) is
PER-DRIVE -- nothing needs to compare one drive's rows to another drive's
rows. So instead of ever assembling the full dataset, this script:

  PASS A (light):  computes the train-only median for imputation by
                    streaming through only the few needed columns, in
                    row-group batches, never holding more than ~1M rows
                    of a handful of columns at once. Exact, not
                    approximated -- uses an exact value-count median.
  PASS B (medium):  reads each of the 43 cleaned quarter files in
                    row-group batches (never a whole quarter at once,
                    since post-clean quarters can still be 15-25M rows),
                    applies the Pass-A imputation, and hash-splits every
                    row by serial_number into one of N drive-partitioned
                    temp files. A drive's rows ALWAYS land in the same
                    partition, regardless of which quarter they came
                    from -- so each temp file ends up holding a drive's
                    COMPLETE history, not a truncated slice.
  PASS C (heavy):   for each of the N drive-partitioned temp files (now
                    small enough to hold in memory -- ~682M/N rows),
                    computes failure date / RUL / the classification
                    target / rolling features / tenure features exactly
                    as the single-frame version would, then writes the
                    result to its own output file. This is the expensive,
                    crash-prone step, so it's fully resumable: a batch
                    whose output file already exists and opens cleanly is
                    skipped on rerun.

OUTPUT IS A DIRECTORY, NOT ONE FILE:
For the same reason Stage 2 kept per-quarter files instead of one combined
file, Stage 3's output here is a directory of per-batch feature files
(features_batch_0000.parquet, ...), not one monolithic file. Combining
these into a single file would reintroduce the exact 300GB problem this
script exists to avoid. 04_temporal_split.py will need to be checked
against this same constraint before it's run -- don't assume it can load
this directory as one frame either.

Usage:
    python3 03_build_targets_and_features.py \
        --in data/processed/cleaned_by_quarter \
        --train_end 2024-03-31 \
        --horizon_days 14 \
        --out_dir data/processed/features_by_batch

Safe to re-run: Pass A/B are skipped if a valid set of partition temp
files from a previous run already exists; Pass C skips any batch whose
output file already exists and opens cleanly.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CORE_SMART_COLS = [
    "smart_5_raw", "smart_187_raw", "smart_188_raw",
    "smart_194_raw", "smart_197_raw", "smart_198_raw",
]
POWER_ON_HOURS_COL = "smart_9_raw"
ROLLING_WINDOWS = [7, 14, 30]

READ_BATCH_ROWS = 1_000_000  # row-group batch size when streaming a parquet file
DEFAULT_N_BATCHES = 60       # ~682M rows / 60 ≈ 11.4M rows per drive-partition,
                              # comfortably inside the range Stage 2 already
                              # proved safe (its largest single quarter was ~24M)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def discover_quarter_files(in_dir: Path) -> list[Path]:
    paths = sorted(in_dir.glob("cleaned_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No cleaned_*.parquet files found in {in_dir}")
    return paths


def get_schema_columns(paths: list[Path]) -> list[str]:
    """Reads just the schema (no data) of the first file to know which
    columns exist -- e.g. which CORE_SMART_COLS actually survived Stage
    2's near-empty-column drop -- without loading any rows."""
    return pq.ParquetFile(paths[0]).schema_arrow.names


def iter_parquet_row_batches(path: Path, columns: list[str] | None, batch_rows: int):
    """Yields the file's rows as pandas DataFrames of at most batch_rows
    rows at a time, via pyarrow's native batch reader -- never loading
    the whole file into memory regardless of its size."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def output_is_valid(path: Path, required_col: str = "failure") -> bool:
    """Same validity check pattern used in Stage 2: a file only counts as
    'done' if it exists AND actually opens without error."""
    if not path.exists():
        return False
    try:
        pq.ParquetFile(path).schema_arrow  # cheap metadata-only check
        # also confirm the expected column is present, catching a
        # truncated write that happened to still have valid footer bytes
        return required_col in pq.ParquetFile(path).schema_arrow.names
    except Exception:
        return False


# ---------------------------------------------------------------------
# PASS A — exact train-only median via streaming value counts
# ---------------------------------------------------------------------

def compute_train_medians_streaming(paths: list[Path], cols: list[str],
                                     train_end: pd.Timestamp) -> pd.Series:
    """Computes an EXACT median (not approximated) for each column, using
    only rows with date <= train_end, without ever holding more than one
    read-batch of a handful of narrow columns in memory. Works by
    accumulating value -> count across every batch/quarter, then finding
    the median position(s) from the merged counts at the end -- memory
    scales with the number of DISTINCT values seen, not the number of
    rows, which is safe for SMART attributes (bounded, far-from-row-count
    cardinality)."""
    if not cols:
        return pd.Series(dtype=float)

    counters = {c: Counter() for c in cols}
    n_seen = {c: 0 for c in cols}
    read_cols = ["date"] + cols

    for p in paths:
        for batch in iter_parquet_row_batches(p, columns=read_cols, batch_rows=READ_BATCH_ROWS):
            train_rows = batch.loc[pd.to_datetime(batch["date"]) <= train_end, cols]
            if train_rows.empty:
                continue
            for c in cols:
                vals = train_rows[c].dropna()
                if vals.empty:
                    continue
                vc = vals.value_counts()
                for v, n in vc.items():
                    counters[c][v] += int(n)
                n_seen[c] += len(vals)
            del train_rows
        del batch

    medians: dict[str, float] = {}
    bad_cols = []
    for c in cols:
        n = n_seen[c]
        if n == 0:
            bad_cols.append(c)
            continue
        sorted_items = sorted(counters[c].items())
        mid_lo = (n - 1) // 2   # 0-indexed position of lower median element
        mid_hi = n // 2         # 0-indexed position of upper median element
        cum = 0
        v_lo = v_hi = None
        for val, cnt in sorted_items:
            cum += cnt
            if v_lo is None and cum - 1 >= mid_lo:
                v_lo = val
            if v_hi is None and cum - 1 >= mid_hi:
                v_hi = val
                break
        medians[c] = (v_lo + v_hi) / 2.0

    if bad_cols:
        raise ValueError(
            f"Could not compute a training-set median for columns {bad_cols} -- "
            f"the training window may be too small or entirely empty for these columns."
        )
    return pd.Series(medians)


# ---------------------------------------------------------------------
# PASS B — drive-partition the full dataset into N temp files
# ---------------------------------------------------------------------

def partition_by_drive(paths: list[Path], tmp_dir: Path, n_batches: int,
                        medians: pd.Series, impute_cols: list[str]) -> list[Path]:
    """Streams every quarter file in row-group batches, applies the
    already-computed imputation medians, and hash-splits rows by
    serial_number into n_batches temp files. A given drive's rows always
    go to the same batch number regardless of which quarter or which
    read-batch they came from, so each temp file ends up holding some
    drives' COMPLETE histories -- never a truncated slice of a drive.

    Memory bound: at most one read-batch (READ_BATCH_ROWS rows, all
    columns) is ever held at once, plus N open writers -- independent of
    total dataset size or any single quarter's size."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths = [tmp_dir / f"batch_{b:04d}.parquet.tmp" for b in range(n_batches)]
    final_paths = [tmp_dir / f"batch_{b:04d}.parquet" for b in range(n_batches)]
    writers: list[pq.ParquetWriter | None] = [None] * n_batches

    total_rows_written = 0
    for p_idx, p in enumerate(paths, start=1):
        print(f"    [PARTITION] ({p_idx}/{len(paths)}) streaming {p.name}...")
        for batch in iter_parquet_row_batches(p, columns=None, batch_rows=READ_BATCH_ROWS):
            if impute_cols:
                present = [c for c in impute_cols if c in batch.columns]
                if present:
                    batch[present] = batch[present].fillna(medians[present])

            serials = batch["serial_number"].astype(str)
            bid = serials.map(lambda s: hash(s) % n_batches).values

            for b in range(n_batches):
                mask = bid == b
                if not mask.any():
                    continue
                piece = batch.loc[mask].reset_index(drop=True)
                table = pa.Table.from_pandas(piece, preserve_index=False)
                if writers[b] is None:
                    writers[b] = pq.ParquetWriter(tmp_paths[b], table.schema)
                writers[b].write_table(table)
                total_rows_written += len(piece)
                del piece, table
            del batch, serials, bid

    for b in range(n_batches):
        if writers[b] is not None:
            writers[b].close()
            tmp_paths[b].replace(final_paths[b])

    print(f"    [PARTITION] done -- {total_rows_written:,} rows distributed across "
          f"{sum(1 for fp in final_paths if fp.exists())} non-empty batch file(s)")
    return [fp for fp in final_paths if fp.exists()]


def partition_pass_already_done(tmp_dir: Path, n_batches: int) -> list[Path] | None:
    """Checks whether a previous run's Pass B fully completed with the
    SAME n_batches setting. Returns the list of valid batch files if so,
    otherwise None (caller should wipe tmp_dir and redo Pass A+B)."""
    manifest_path = tmp_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if manifest.get("n_batches") != n_batches:
        return None

    candidates = [tmp_dir / f"batch_{b:04d}.parquet" for b in range(n_batches)]
    valid = [p for p in candidates if output_is_valid(p, required_col="serial_number")]
    # It's fine for some batches to be legitimately empty/absent (a hash
    # bucket got no drives) -- but we can't tell "genuinely empty" apart
    # from "never got written" from files alone, so we trust the manifest
    # here rather than re-deriving it.
    if len(valid) != manifest.get("n_nonempty_batches"):
        return None
    return valid


# ---------------------------------------------------------------------
# PASS C — per-drive feature engineering (same logic as the single-frame
# version, just invoked once per drive-partitioned batch instead of once
# on the whole dataset)
# ---------------------------------------------------------------------

def add_failure_date_and_rul(df: pd.DataFrame, horizon_days: int,
                              include_failure_day: bool = False) -> pd.DataFrame:
    """See module docstring -- unchanged logic from the single-frame
    version. Safe to run per-batch because every drive's rows are
    guaranteed (by the Pass-B hash partition) to live entirely within one
    batch, so this still sees each drive's FULL history."""
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

    fail_dates = (
        df.loc[df["failure"] == 1]
        .groupby("serial_number", as_index=False)["date"]
        .max()
        .rename(columns={"date": "failure_date"})
    )
    df = df.merge(fail_dates, on="serial_number", how="left")

    df["rul_days"] = (df["failure_date"] - df["date"]).dt.days
    df["is_censored"] = df["failure_date"].isna().astype(int)

    lower_bound = 0 if include_failure_day else 1
    df[f"failure_within_{horizon_days}d"] = (
        (~df["is_censored"].astype(bool))
        & (df["rul_days"] >= lower_bound)
        & (df["rul_days"] <= horizon_days)
    ).astype(int)

    df = df.drop(columns=["failure_date"])
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Unchanged from the single-frame version -- see module docstring."""
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

    for col in cols:
        if col not in df.columns:
            continue

        for w in ROLLING_WINDOWS:
            window_str = f"{w}D"

            def _time_roll(g, window_str=window_str, col=col):
                idxd = g.set_index("date")
                return pd.DataFrame({
                    "date": g["date"].values,
                    "roll_mean": idxd[col].rolling(window_str, min_periods=1).mean().values,
                    "roll_std": idxd[col].rolling(window_str, min_periods=2).std().values,
                })

            rolled = (
                df[["serial_number", "date", col]]
                .groupby("serial_number", group_keys=True)
                .apply(_time_roll)
                .reset_index(level=0)
                .reset_index(drop=True)
            )

            df = df.merge(
                rolled.rename(columns={
                    "roll_mean": f"{col}_roll{w}_mean",
                    "roll_std": f"{col}_roll{w}_std",
                }),
                on=["serial_number", "date"], how="left", validate="one_to_one",
            )

        df[f"{col}_delta"] = df.groupby("serial_number", group_keys=False)[col].diff().fillna(0.0)

    return df


def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    install_date = df.groupby("serial_number")["date"].transform("min")
    df["days_since_install"] = (df["date"] - install_date).dt.days

    if POWER_ON_HOURS_COL in df.columns:
        df["power_on_years"] = df[POWER_ON_HOURS_COL] / 8760.0
    else:
        df["power_on_years"] = np.nan

    present_error_cols = [c for c in ["smart_187_raw", "smart_197_raw", "smart_198_raw"] if c in df.columns]
    df["error_total"] = df[present_error_cols].sum(axis=1) if present_error_cols else 0.0

    if "smart_194_raw" in df.columns:
        normal_temp = 30.0
        df["temp_stress"] = (df["smart_194_raw"] - normal_temp).clip(lower=0)
    else:
        df["temp_stress"] = np.nan

    return df


def process_one_batch(batch_path: Path, out_path: Path, horizon_days: int, present_cols: list[str]) -> dict:
    df = pd.read_parquet(batch_path)
    df["date"] = pd.to_datetime(df["date"])

    df = add_failure_date_and_rul(df, horizon_days)
    df = add_rolling_features(df, present_cols)
    df = add_tenure_features(df)

    tmp_path = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    target_col = f"failure_within_{horizon_days}d"
    stats = {
        "rows": len(df),
        "n_censored": int(df["is_censored"].sum()),
        "positive_rate": float(df[target_col].mean()),
    }
    del df
    return stats


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build targets and features, drive-partitioned to stay memory-safe")
    parser.add_argument("--in", dest="in_dir", type=str, required=True,
                         help="Directory of cleaned_*.parquet files from Stage 2")
    parser.add_argument("--train_end", type=str, required=True,
                         help="Last date (inclusive) belonging to the training split -- "
                              "used ONLY to compute the imputation statistic without leakage. "
                              "The actual physical split still happens in 04_temporal_split.py.")
    parser.add_argument("--horizon_days", type=int, default=14)
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Output directory -- one features_batch_XXXX.parquet file per "
                              "drive-partition is written here (NOT a single combined file; "
                              "see module docstring for why).")
    parser.add_argument("--n_batches", type=int, default=DEFAULT_N_BATCHES)
    parser.add_argument("--tmp_dir", type=str, default=None,
                         help="Where Pass B's intermediate drive-partition files live. "
                              "Defaults to <out_dir>/_tmp_partitions")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else out_dir / "_tmp_partitions"

    train_end_ts = pd.Timestamp(args.train_end)
    paths = discover_quarter_files(in_dir)
    print(f"[DISCOVER] {len(paths)} cleaned quarter file(s) in {in_dir}")

    schema_cols = get_schema_columns(paths)
    present_cols = [c for c in CORE_SMART_COLS if c in schema_cols]
    missing_cols = [c for c in CORE_SMART_COLS if c not in schema_cols]
    if missing_cols:
        print(f"NOTE: expected SMART columns not present, skipping: {missing_cols}")

    # ---- Pass A + B: resume if a matching partition already exists ----
    batch_paths = partition_pass_already_done(tmp_dir, args.n_batches)
    if batch_paths is not None:
        print(f"[RESUME] Found a valid drive-partition from a previous run "
              f"({len(batch_paths)} batch file(s), n_batches={args.n_batches}) -- "
              f"skipping Pass A (medians) and Pass B (partitioning).")
    else:
        if tmp_dir.exists():
            print(f"[RESUME] No valid matching partition found -- clearing {tmp_dir} and redoing Pass A+B.")
            shutil.rmtree(tmp_dir)

        print(f"[PASS A] Computing train-only medians for {present_cols} "
              f"(rows with date <= {train_end_ts.date()})...")
        medians = compute_train_medians_streaming(paths, present_cols, train_end_ts)
        print(f"[PASS A] medians: {medians.to_dict()}")

        print(f"[PASS B] Partitioning full dataset into {args.n_batches} drive-safe batch(es)...")
        batch_paths = partition_by_drive(paths, tmp_dir, args.n_batches, medians, present_cols)

        manifest = {"n_batches": args.n_batches, "n_nonempty_batches": len(batch_paths)}
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest))

    # ---- Pass C: per-batch feature engineering, resumable per batch ----
    print(f"[PASS C] Building targets/features for {len(batch_paths)} drive-partition batch(es)...")
    total_rows = 0
    total_censored = 0
    positive_rates = []

    for i, bp in enumerate(sorted(batch_paths), start=1):
        out_path = out_dir / f"features_{bp.stem}.parquet"
        if output_is_valid(out_path, required_col=f"failure_within_{args.horizon_days}d"):
            print(f"[{i}/{len(batch_paths)}] [{bp.name}] SKIPPED -- already built at {out_path}")
            continue

        print(f"[{i}/{len(batch_paths)}] [{bp.name}] processing...")
        stats = process_one_batch(bp, out_path, args.horizon_days, present_cols)
        total_rows += stats["rows"]
        total_censored += stats["n_censored"]
        positive_rates.append(stats["positive_rate"])
        print(f"[{i}/{len(batch_paths)}] [{bp.name}] {stats['rows']:,} rows written -> {out_path} "
              f"(positive rate: {stats['positive_rate']:.4%})")

    print(f"\n[DONE] Feature files written to {out_dir}")
    if positive_rates:
        print(f"       this run's new rows: {total_rows:,} | censored rows: {total_censored:,} | "
              f"avg positive rate across newly-built batches: {np.mean(positive_rates):.4%}")
    print(f"       NOTE: output is a DIRECTORY of per-batch files, not one combined file -- "
          f"check 04_temporal_split.py's memory approach before running it on this directory.")


if __name__ == "__main__":
    main()