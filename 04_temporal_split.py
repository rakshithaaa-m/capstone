"""
Stage 4 — Temporal Train / Validation / Test Split (MEMORY-SAFE, BATCH-WISE)

The original 04_temporal_split.py expected ONE file (features_full.parquet)
and called pd.read_parquet() on it, then held the entire dataset in memory
to compute the three-way split. Stage 3 deliberately never produces that one
file -- it writes 60 separate features_batch_XXXX.parquet files instead,
specifically because combining them into a single in-memory dataframe would
reintroduce the ~300GB RAM requirement that broke the original Stage 3 on
this hardware. Loading everything here would hit the exact same wall.

This version applies the same pattern Stage 3's Pass C used: process one
batch at a time, and write each batch's contribution to train/val/test
incrementally (via pyarrow.parquet.ParquetWriter) rather than accumulating
whole splits in memory and writing once at the end.

Because every batch is a self-contained, drive-partitioned slice with its
own date column, the train/val/test boundary logic is IDENTICAL per batch
to what the original script did on the full dataframe -- a row's split
membership only depends on its own date, never on other rows. So doing this
batch-by-batch produces exactly the same train/val/test assignment as doing
it all at once would have.

OUTPUT: data/processed/features_train.parquet, features_val.parquet,
features_test.parquet -- same three files the original script produced,
just built incrementally. These ARE single combined files (unlike Stage 3's
output) because after the split, train/val/test are each roughly 1/3 to
1/60th-ish the size of the full dataset depending on your date boundaries,
and typical model training APIs (LightGBM, sklearn) expect one file or
in-memory table per split. If any one split is still too large to load at
once on this hardware, see the note on chunked/incremental training at the
bottom of this docstring.

Usage:
    python3 04_temporal_split.py \
        --in_dir data/processed/features_by_batch \
        --train_end 2024-03-31 \
        --val_end 2024-04-30 \
        --out_dir data/processed
"""
import argparse
import gc
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROW_CHUNK_SIZE = 500_000  # rows read at a time from each batch file -- keeps
                           # every in-memory array well under the point where
                           # a single contiguous allocation can fail even
                           # when overall free memory looks sufficient.


def discover_batch_files(in_dir: Path) -> list[Path]:
    files = sorted(in_dir.glob("features_batch_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No features_batch_*.parquet files found in {in_dir}. "
            f"Did Stage 3 (03_build_targets_and_features.py) finish?"
        )
    return files


class IncrementalSplitWriter:
    """Writes rows to train/val/test parquet files incrementally, one batch
    at a time, and keeps running totals for the summary stats the original
    script printed -- so nothing is ever fully loaded in memory."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.writers: dict[str, pq.ParquetWriter | None] = {"train": None, "val": None, "test": None}
        self.schemas: dict[str, pa.Schema | None] = {"train": None, "val": None, "test": None}
        self.paths = {
            "train": out_dir / "features_train.parquet",
            "val": out_dir / "features_val.parquet",
            "test": out_dir / "features_test.parquet",
        }
        self.tmp_paths = {k: p.with_suffix(".parquet.tmp") for k, p in self.paths.items()}
        self.row_counts = {"train": 0, "val": 0, "test": 0}
        self.failure_sums = {"train": 0, "val": 0, "test": 0}
        self.censored_sums = {"train": 0, "val": 0, "test": 0}
        self.horizon_sums = {"train": 0, "val": 0, "test": 0}
        self.date_min = {"train": None, "val": None, "test": None}
        self.date_max = {"train": None, "val": None, "test": None}
        self.horizon_col = None

    def write_part(self, name: str, part: pd.DataFrame) -> None:
        if len(part) == 0:
            return

        if self.horizon_col is None:
            horizon_cols = [c for c in part.columns if c.startswith("failure_within_")]
            self.horizon_col = horizon_cols[0] if horizon_cols else None

        table = pa.Table.from_pandas(part, preserve_index=False)
        if self.writers[name] is None:
            self.writers[name] = pq.ParquetWriter(self.tmp_paths[name], table.schema)
            self.schemas[name] = table.schema
        elif not table.schema.equals(self.schemas[name]):
            # Safety net: if some other column ever drifts in type across
            # batches (beyond the days_since_install case already handled
            # above), reconcile it against the schema already committed to
            # this file rather than crashing the whole run.
            table = table.cast(self.schemas[name], safe=False)
        self.writers[name].write_table(table)

        self.row_counts[name] += len(part)
        self.failure_sums[name] += int(part["failure"].sum())
        if "is_censored" in part.columns:
            self.censored_sums[name] += int(part["is_censored"].sum())
        if self.horizon_col:
            self.horizon_sums[name] += int(part[self.horizon_col].sum())

        pmin, pmax = part["date"].min(), part["date"].max()
        self.date_min[name] = pmin if self.date_min[name] is None else min(self.date_min[name], pmin)
        self.date_max[name] = pmax if self.date_max[name] is None else max(self.date_max[name], pmax)

    def close_and_report(self) -> None:
        for name, writer in self.writers.items():
            if writer is not None:
                writer.close()
                self.tmp_paths[name].replace(self.paths[name])  # atomic write

        print()
        for name in ["train", "val", "test"]:
            n = self.row_counts[name]
            if n == 0:
                print(f"{name:5s}: EMPTY")
                continue
            date_range = f"{self.date_min[name].date()} to {self.date_max[name].date()}"
            raw_rate = self.failure_sums[name] / n
            censored_rate = self.censored_sums[name] / n
            horizon_rate = (self.horizon_sums[name] / n) if self.horizon_col else float("nan")
            print(f"{name:5s}: {n:>12,} rows | {date_range}")
            print(f"       raw failure rate: {raw_rate:.6%}  |  "
                  f"{self.horizon_col or 'N-day target'} positive rate: {horizon_rate:.6%}  |  "
                  f"censored rate: {censored_rate:.4%}")

        if self.row_counts["val"] == 0 or self.row_counts["test"] == 0:
            print("WARNING: val or test split is empty -- check your date boundaries.")
        if self.failure_sums["train"] == 0:
            print("WARNING: zero failures in the training split -- the model will have nothing to learn from.")
        if self.horizon_col and self.horizon_sums["train"] == 0:
            print(f"WARNING: zero positive '{self.horizon_col}' examples in training -- push --train_end later "
                  f"so at least some failures fall within the horizon before the cutoff.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal split of Stage 3's batch-partitioned feature files")
    parser.add_argument("--in_dir", type=str, default="data/processed/features_by_batch",
                         help="Directory of Stage 3's features_batch_*.parquet files")
    parser.add_argument("--train_end", type=str, required=True,
                         help="Must match the --train_end used in Stage 3, so the imputation "
                              "statistic and the physical split boundary agree.")
    parser.add_argument("--val_end", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_end_ts = pd.Timestamp(args.train_end)
    val_end_ts = pd.Timestamp(args.val_end)
    if val_end_ts <= train_end_ts:
        raise ValueError(f"--val_end ({val_end_ts.date()}) must be after --train_end ({train_end_ts.date()})")

    batch_files = discover_batch_files(in_dir)
    print(f"[DISCOVER] {len(batch_files)} batch file(s) in {in_dir}")

    writer = IncrementalSplitWriter(out_dir)

    for i, path in enumerate(batch_files, start=1):
        print(f"[{i}/{len(batch_files)}] [{path.stem}] processing...")

        # Stream this batch file in small row-chunks (like Stage 3's Pass B)
        # instead of loading all ~11M rows at once. Even after freeing
        # train/val/test one at a time, a single ~11M-row batch still needs
        # one big CONTIGUOUS array per split -- and that can fail even when
        # total free memory looks sufficient, if it's fragmented. Smaller
        # chunks keep every allocation small enough to reliably succeed.
        parquet_file = pq.ParquetFile(path)
        for chunk_num, row_group_batch in enumerate(
            parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE), start=1
        ):
            df = row_group_batch.to_pandas()
            df["date"] = pd.to_datetime(df["date"])

            # Stage 3 wrote each of the 60 batch files independently. Some
            # batches happened to have no nulls in days_since_install (and
            # got stored as int64), others had at least one null (and got
            # stored as double) -- this is invisible until Stage 4 tries to
            # write batches with different types into the same continuous
            # output file. Force any originally-int64 column to float64 on
            # every chunk, unconditionally, so every batch agrees no matter
            # which source file it came from. Harmless for training (LightGBM
            # and friends treat these identically) and avoids ever hitting
            # this mismatch again regardless of which batch shows up next.
            int_cols = df.select_dtypes(include="int64").columns
            if len(int_cols) > 0:
                df[int_cols] = df[int_cols].astype("float64")

            train = df[df["date"] <= train_end_ts]
            writer.write_part("train", train)
            del train

            val = df[(df["date"] > train_end_ts) & (df["date"] <= val_end_ts)]
            writer.write_part("val", val)
            del val

            test = df[df["date"] > val_end_ts]
            writer.write_part("test", test)
            del test

            del df
            if chunk_num % 5 == 0:
                gc.collect()

        gc.collect()

    writer.close_and_report()

    print(f"\n[DONE] Stage 4 complete. Splits written to {out_dir}/features_train.parquet, "
          f"features_val.parquet, features_test.parquet")


if __name__ == "__main__":
    main()