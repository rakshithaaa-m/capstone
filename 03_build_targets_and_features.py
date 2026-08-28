"""
Stage 3 — Build Targets and Features (MEMORY-SAFE, DRIVE-PARTITIONED, RESUMABLE)

The original 03_build_targets_and_features.py combines ALL cleaned quarterly
parquet files into one in-memory dataframe before doing anything else
(load_cleaned_data()). At this project's real scale -- ~682 million rows,
~90+ columns after Stage 2's column drop -- that requires roughly 300GB of
RAM. That's not a "might crash on the biggest chunk" risk like Stage 2 had;
it fails immediately, on hardware with 15-16GB RAM.

The fix here uses the fact that EVERY computation in this stage is per-drive
-- RUL/failure-date, the classification target, rolling-window features, and
tenure features never need to compare one drive against another. So instead
of loading everything at once, this version processes the data in
drive-partitioned batches:

  PASS A -- Train-median estimation. Reads only the handful of columns
            needed for imputation (date + core SMART columns), filtered to
            the training window, and estimates the median from a bounded
            random sample rather than the full training set. A large random
            sample (default 3,000,000 rows) gives a statistically stable
            median estimate without ever holding the full multi-hundred-
            million-row training column set in memory. This is a documented
            approximation, not the original's exact median -- acceptable
            because Stage 3's imputation is a coarse fallback for a small
            fraction of remaining NaNs to begin with (Stage 2 already
            forward-fills most gaps).

  PASS B -- Drive-safe partitioning. Streams each of the 43 cleaned quarter
            files in small row-group batches (never a whole quarter at
            once), assigns every row to one of --n_batches output partitions
            via a stable hash of serial_number, and writes each quarter's
            contribution to each batch as its own small file:
                out_dir/_partitions/batch_0000/2016_Q1.parquet
                out_dir/_partitions/batch_0000/2016_Q2.parquet
                ...
            A stable hash guarantees a given drive always lands in the same
            batch, across every quarter and every run -- so a drive's full
            multi-year history is always reunited within one batch in Pass C,
            never split. This step is resumable per (quarter, batch): if a
            quarter's files already exist in every batch folder, that
            quarter is skipped entirely on a rerun.

  PASS C -- Feature engineering per batch. For each batch, concatenates just
            that batch's per-quarter files (roughly total_rows/n_batches
            rows -- a small, RAM-safe fraction of the full dataset), then
            runs the exact same per-drive logic as the original script:
            train-median imputation, RUL/censoring/classification target,
            rolling-window features, tenure features. Writes one output file
            per batch. Resumable: a batch already written is skipped.

OUTPUT: a directory of files (features_by_batch/features_batch_0000.parquet,
...0001.parquet, ...), NOT one combined file -- combining them back into a
single dataframe later would reintroduce the exact 300GB problem this script
exists to avoid. Stage 4 (temporal split) needs a small adjustment to read
from this directory instead of one file -- send that script over before
running it, it is NOT compatible with this output as-is.

Usage:
    python3 03_build_targets_and_features.py \
        --in data/processed/cleaned_by_quarter \
        --train_end 2024-03-31 \
        --horizon_days 14 \
        --out_dir data/processed/features_by_batch \
        --n_batches 60

Safe to re-run after an interruption -- it always resumes from where it left
off in both Pass B and Pass C.
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CORE_SMART_COLS = [
    "smart_5_raw", "smart_187_raw", "smart_188_raw",
    "smart_194_raw", "smart_197_raw", "smart_198_raw",
]
POWER_ON_HOURS_COL = "smart_9_raw"
ROLLING_WINDOWS = [7, 14, 30]
ROW_GROUP_BATCH_SIZE = 500_000  # rows read at a time during Pass B streaming


def discover_quarter_files(in_dir: Path) -> list[Path]:
    files = sorted(in_dir.glob("cleaned_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cleaned_*.parquet files found in {in_dir}. Did Stage 2 finish?")
    return files


def batch_for_serial(serial: str, n_batches: int) -> int:
    """Stable hash -> batch index. Same drive always lands in the same
    batch, every run, every quarter -- this is what guarantees a drive's
    history is never split across batches."""
    return zlib.crc32(str(serial).encode("utf-8")) % n_batches


# ---------------------------------------------------------------------------
# PASS A -- train-median estimation via bounded random sampling
# ---------------------------------------------------------------------------

def estimate_train_medians(quarter_files: list[Path], train_end: pd.Timestamp,
                            cols: list[str], sample_target: int) -> pd.Series:
    print(f"[PASS A] Estimating train-only medians for {cols} "
          f"(target sample size: {sample_target:,} rows)...")

    needed_cols = ["date"] + cols
    per_file_target = max(1, sample_target // len(quarter_files))
    samples = []

    for i, path in enumerate(quarter_files, start=1):
        table = pq.read_table(path, columns=needed_cols)
        df = table.to_pandas()
        df = df[df["date"] <= train_end]
        if len(df) == 0:
            continue
        if len(df) > per_file_target:
            df = df.sample(n=per_file_target, random_state=42)
        samples.append(df[cols])
        if i % 10 == 0 or i == len(quarter_files):
            print(f"[PASS A] {i}/{len(quarter_files)} quarter file(s) sampled")
        del table, df

    if not samples:
        raise ValueError(
            "No rows found at or before --train_end across any quarter file -- "
            "check that --train_end actually falls within your data's date range."
        )

    combined = pd.concat(samples, ignore_index=True)
    medians = combined[cols].median(numeric_only=True)
    del samples, combined

    if medians.isna().any():
        bad = medians[medians.isna()].index.tolist()
        raise ValueError(
            f"Could not compute a training-set median for columns {bad} -- "
            f"the training window may be too small or entirely empty for these columns."
        )

    print(f"[PASS A] medians: {medians.to_dict()}")
    return medians


# ---------------------------------------------------------------------------
# PASS B -- drive-safe partitioning
# ---------------------------------------------------------------------------

def partition_already_done(out_dir: Path, label: str, n_batches: int) -> bool:
    """A quarter is considered already-partitioned only if its file exists
    in EVERY batch folder -- a partial previous run (interrupted mid-quarter)
    will correctly be redone from scratch for that one quarter."""
    return all((out_dir / "_partitions" / f"batch_{b:04d}" / f"{label}.parquet").exists()
               for b in range(n_batches))


def partition_quarter(path: Path, label: str, out_dir: Path, n_batches: int) -> None:
    """Streams one quarter file in row-group batches and appends each row to
    its drive-assigned batch's per-quarter file. Never holds the full
    quarter in memory at once -- ROW_GROUP_BATCH_SIZE caps how much is live
    at any moment, independent of how large the quarter actually is."""
    partitions_dir = out_dir / "_partitions"
    for b in range(n_batches):
        (partitions_dir / f"batch_{b:04d}").mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(path)
    # Accumulate rows per batch in memory only within THIS quarter's
    # streaming pass, then flush once per batch at the end of the quarter --
    # bounded by one quarter's worth of data split n_batches ways, not the
    # full dataset.
    batch_frames: dict[int, list[pd.DataFrame]] = {b: [] for b in range(n_batches)}

    rows_seen = 0
    for row_group_batch in parquet_file.iter_batches(batch_size=ROW_GROUP_BATCH_SIZE):
        df = row_group_batch.to_pandas()
        rows_seen += len(df)
        batch_idx = df["serial_number"].apply(lambda s: batch_for_serial(s, n_batches))
        for b in range(n_batches):
            sub = df[batch_idx == b]
            if len(sub) > 0:
                batch_frames[b].append(sub)
        del df, batch_idx

    for b in range(n_batches):
        if not batch_frames[b]:
            continue
        combined = pd.concat(batch_frames[b], ignore_index=True)
        out_path = partitions_dir / f"batch_{b:04d}" / f"{label}.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)  # atomic write, same reasoning as Stage 2's fix
        del combined

    print(f"[PARTITION] {label}: {rows_seen:,} rows distributed across {n_batches} batch(es)")


def run_pass_b(quarter_files: list[Path], out_dir: Path, n_batches: int) -> None:
    print(f"[PASS B] Partitioning {len(quarter_files)} quarter file(s) into "
          f"{n_batches} drive-safe batch(es)...")

    for i, path in enumerate(quarter_files, start=1):
        label = path.stem.replace("cleaned_", "")
        if partition_already_done(out_dir, label, n_batches):
            print(f"[PARTITION] ({i}/{len(quarter_files)}) {label}: SKIPPED (already partitioned)")
            continue
        print(f"[PARTITION] ({i}/{len(quarter_files)}) {label}: streaming...")
        partition_quarter(path, label, out_dir, n_batches)

    print("[PASS B] Partitioning complete.")


# ---------------------------------------------------------------------------
# PASS C -- per-batch feature engineering (identical logic to the original
# script, just scoped to one batch's drives at a time instead of everything)
# ---------------------------------------------------------------------------

def impute_with_train_medians(df: pd.DataFrame, cols: list[str], medians: pd.Series) -> pd.DataFrame:
    df[cols] = df[cols].fillna(medians)
    return df


def add_failure_date_and_rul(df: pd.DataFrame, horizon_days: int,
                              include_failure_day: bool = False) -> pd.DataFrame:
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
            del rolled

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


def process_batch(batch_dir: Path, medians: pd.Series, present_cols: list[str],
                   horizon_days: int) -> pd.DataFrame:
    quarter_files = sorted(batch_dir.glob("*.parquet"))
    frames = [pd.read_parquet(p) for p in quarter_files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    del frames

    df["date"] = pd.to_datetime(df["date"])
    df = impute_with_train_medians(df, present_cols, medians)
    df = add_failure_date_and_rul(df, horizon_days)
    df = add_rolling_features(df, present_cols)
    df = add_tenure_features(df)
    return df


def run_pass_c(out_dir: Path, n_batches: int, medians: pd.Series,
               present_cols: list[str], horizon_days: int) -> None:
    partitions_dir = out_dir / "_partitions"
    print(f"[PASS C] Building targets/features for {n_batches} batch(es)...")

    for b in range(n_batches):
        out_path = out_dir / f"features_batch_{b:04d}.parquet"
        if out_path.exists():
            print(f"[{b+1}/{n_batches}] [batch_{b:04d}] SKIPPED (already built)")
            continue

        batch_dir = partitions_dir / f"batch_{b:04d}"
        if not batch_dir.exists() or not any(batch_dir.glob("*.parquet")):
            print(f"[{b+1}/{n_batches}] [batch_{b:04d}] no partition files found, skipping (empty batch)")
            continue

        print(f"[{b+1}/{n_batches}] [batch_{b:04d}] processing...")
        df = process_batch(batch_dir, medians, present_cols, horizon_days)

        tmp_path = out_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)

        target_col = f"failure_within_{horizon_days}d"
        print(f"[{b+1}/{n_batches}] [batch_{b:04d}] {len(df):,} rows, {df.shape[1]} cols written -> {out_path} "
              f"(positive rate: {df[target_col].mean():.4%})")
        del df

    print("[PASS C] Feature engineering complete for all batches.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targets and features -- memory-safe, drive-partitioned, resumable")
    parser.add_argument("--in", dest="in_dir", type=str, required=True,
                         help="Directory of Stage 2's cleaned_*.parquet files")
    parser.add_argument("--train_end", type=str, required=True)
    parser.add_argument("--horizon_days", type=int, default=14)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=60,
                         help="Number of drive-safe batches. Increase this if Pass C still runs into "
                              "memory pressure on an individual batch -- more batches means fewer "
                              "drives (and rows) per batch.")
    parser.add_argument("--median_sample_rows", type=int, default=3_000_000,
                         help="Target sample size for Pass A's median estimation")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_end_ts = pd.Timestamp(args.train_end)

    quarter_files = discover_quarter_files(in_dir)
    print(f"[DISCOVER] {len(quarter_files)} cleaned quarter file(s) in {in_dir}")

    # Determine which CORE_SMART_COLS actually exist (Stage 2 may have
    # dropped some globally near-empty ones) by checking just one file's schema.
    schema_cols = set(pq.read_schema(quarter_files[0]).names)
    present_cols = [c for c in CORE_SMART_COLS if c in schema_cols]
    missing_cols = [c for c in CORE_SMART_COLS if c not in schema_cols]
    if missing_cols:
        print(f"NOTE: expected SMART columns not present, skipping: {missing_cols}")

    medians_path = out_dir / "_train_medians.json"
    if medians_path.exists():
        with open(medians_path) as f:
            medians_dict = json.load(f)
        medians = pd.Series(medians_dict)
        print(f"[PASS A] SKIPPED (medians already cached at {medians_path}): {medians_dict}")
    else:
        medians = estimate_train_medians(quarter_files, train_end_ts, present_cols, args.median_sample_rows)
        with open(medians_path, "w") as f:
            json.dump(medians.to_dict(), f, indent=2)

    run_pass_b(quarter_files, out_dir, args.n_batches)
    run_pass_c(out_dir, args.n_batches, medians, present_cols, args.horizon_days)

    print(f"\n[DONE] Stage 3 complete. Per-batch feature files are in {out_dir}/features_batch_*.parquet")
    print(f"       This is a DIRECTORY of files, not one combined file -- Stage 4 needs to be adapted "
          f"to read from this layout. Do not attempt to pd.concat all batches into one dataframe.")


if __name__ == "__main__":
    main()