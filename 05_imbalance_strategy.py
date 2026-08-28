"""
Stage 5 — Imbalance Strategy (MEMORY-SAFE, STREAMED)

Two things happen here, and only here:
  1. Define and save a cost matrix (relative cost of a missed failure vs.
     a false alarm) as JSON, used later by Stage 6+/9 to pick decision
     thresholds and by Stage 9's risk-tier calibration.
  2. Produce an undersampled TRAINING set only (never touch val/test —
     they must reflect the real-world failure rate to give honest metrics).

The original version called pd.read_parquet() on the full training file to
do this in memory. On this hardware that file is ~23GB on disk (and
considerably larger once loaded as a pandas dataframe) -- given the memory
failures already hit in Stage 4 on far smaller allocations, loading it
whole here would almost certainly fail the same way. This version never
holds more than one small chunk in memory at a time, using a two-pass
streaming approach:

  PASS 1 -- reads ONLY the label column (cheap, low memory) across the
            whole file to get the EXACT count of minority (failure=1) and
            majority (failure=0) rows. This tells us exactly how many
            majority rows to keep to hit the target ratio, without ever
            loading the full dataset.

  PASS 2 -- streams all columns in small row-chunks. Every minority row is
            kept. Majority rows are kept via random Bernoulli sampling at
            a keep-probability computed from Pass 1's exact counts, so the
            final majority count lands very close to the target (not
            exact to the row, unlike the original's df.sample(n=...) --
            an acceptable, documented tradeoff for never materializing the
            full dataset). Written directly to disk as it goes.

Row order in the output is NOT globally shuffled (unlike the original,
which did df.sample(frac=1.0) at the end) -- shuffling requires holding
everything in memory at once, defeating the point of streaming. This is
safe for LightGBM: it doesn't process rows sequentially the way an SGD-style
or sequence model would, so row order has no effect on training. If you
later feed this data to something order-sensitive, shuffle it then.

Usage:
    python3 05_imbalance_strategy.py \
        --in data/processed/features_train.parquet \
        --out data/processed/train_resampled.parquet \
        --cost_out data/processed/cost_matrix.json \
        --ratio 100 \
        --fn_cost 100 --fp_cost 1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROW_CHUNK_SIZE = 500_000  # rows read at a time in Pass 2 -- keeps every
                           # in-memory chunk small regardless of the full
                           # file's size.


def build_cost_matrix(fn_cost: float, fp_cost: float) -> dict:
    """fn_cost: cost of missing a real failure (e.g. downtime + data-recovery
    cost, in arbitrary units). fp_cost: cost of a false alarm (e.g. a
    technician-hours cost to check a healthy drive). The ratio between them
    is what actually matters for threshold selection, not the absolute
    numbers — pick numbers you can defend/justify in your report, e.g. from
    a rough SLA-penalty vs technician-hourly-rate estimate."""
    if fn_cost <= 0 or fp_cost <= 0:
        raise ValueError("Costs must be positive.")
    return {
        "false_negative_cost": fn_cost,
        "false_positive_cost": fp_cost,
        "cost_ratio_fn_to_fp": fn_cost / fp_cost,
        "note": (
            "Used to select the decision threshold that minimizes "
            "expected cost = FN_count * false_negative_cost + FP_count * false_positive_cost, "
            "evaluated on the validation set in Stage 6/9, not on test."
        ),
    }


def count_minority_majority(path: Path, label_col: str) -> tuple[int, int]:
    """Pass 1 -- scans ONLY the label column to get exact minority/majority
    counts without ever loading the full set of feature columns."""
    parquet_file = pq.ParquetFile(path)
    if label_col not in parquet_file.schema_arrow.names:
        raise ValueError(f"'{label_col}' column not found in {path}.")

    n_minority = 0
    n_majority = 0
    for batch in parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE, columns=[label_col]):
        labels = batch.column(0).to_numpy(zero_copy_only=False)
        n_minority += int((labels == 1).sum())
        n_majority += int((labels == 0).sum())
    return n_minority, n_majority


def stream_undersample(path: Path, out_path: Path, label_col: str,
                        n_minority: int, n_majority: int, ratio: float,
                        random_state: int = 42) -> tuple[int, int]:
    """Pass 2 -- streams all columns in chunks, keeps every minority row,
    and Bernoulli-samples majority rows at the keep-probability needed to
    hit the target ratio. Writes directly to out_path as it goes."""
    if n_minority == 0:
        raise ValueError(
            "No positive (failure=1) rows found in the training set — check that Stage 3's "
            "date boundaries actually leave failures in the training split."
        )

    majority_target = min(n_majority, int(n_minority * ratio))
    keep_prob = majority_target / n_majority if n_majority > 0 else 0.0
    print(f"[PASS 2] minority={n_minority:,}  majority={n_majority:,}  "
          f"majority_target={majority_target:,}  keep_prob={keep_prob:.6f}")

    rng = np.random.default_rng(random_state)
    parquet_file = pq.ParquetFile(path)

    writer = None
    tmp_path = out_path.with_suffix(".parquet.tmp")
    total_minority_written = 0
    total_majority_written = 0

    for chunk_num, row_group_batch in enumerate(
        parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE), start=1
    ):
        df = row_group_batch.to_pandas()

        minority_rows = df[df[label_col] == 1]
        majority_rows = df[df[label_col] == 0]

        if len(majority_rows) > 0 and keep_prob > 0:
            mask = rng.random(len(majority_rows)) < keep_prob
            majority_sampled = majority_rows[mask]
        else:
            majority_sampled = majority_rows.iloc[0:0]

        out_chunk = pd.concat([minority_rows, majority_sampled], ignore_index=True)
        total_minority_written += len(minority_rows)
        total_majority_written += len(majority_sampled)

        if len(out_chunk) > 0:
            table = pa.Table.from_pandas(out_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)

        del df, minority_rows, majority_rows, majority_sampled, out_chunk

        if chunk_num % 20 == 0:
            print(f"[PASS 2] {chunk_num} chunk(s) processed — "
                  f"{total_minority_written:,} minority + {total_majority_written:,} majority written so far")

    if writer is not None:
        writer.close()
        tmp_path.replace(out_path)

    return total_minority_written, total_majority_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cost matrix and undersample the training set (streamed)")
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--cost_out", type=str, default="data/processed/cost_matrix.json")
    parser.add_argument("--ratio", type=float, default=100.0,
                         help="majority:minority ratio to keep, e.g. 100 means ~100 healthy rows per failure row")
    parser.add_argument("--fn_cost", type=float, default=100.0)
    parser.add_argument("--fp_cost", type=float, default=1.0)
    parser.add_argument("--label_col", type=str, default="failure_within_14d",
                         help="Which column to undersample against. Use the actual "
                              "classification target your model trains on (e.g. "
                              "'failure_within_14d' from Stage 3) — NOT the raw 'failure' "
                              "column, which is 1 only on the exact failure day itself.")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cost_matrix = build_cost_matrix(args.fn_cost, args.fp_cost)
    cost_out_path = Path(args.cost_out)
    cost_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_out_path, "w") as f:
        json.dump(cost_matrix, f, indent=2)
    print(f"Cost matrix -> {cost_out_path}: {cost_matrix}")

    print(f"[PASS 1] Counting minority/majority rows in {in_path}...")
    n_minority, n_majority = count_minority_majority(in_path, args.label_col)
    total_rows = n_minority + n_majority
    before_rate = n_minority / total_rows if total_rows > 0 else float("nan")
    print(f"[PASS 1] total={total_rows:,}  minority={n_minority:,}  majority={n_majority:,}  "
          f"failure rate={before_rate:.4%}")

    minority_written, majority_written = stream_undersample(
        in_path, out_path, args.label_col, n_minority, n_majority, args.ratio
    )

    total_written = minority_written + majority_written
    after_rate = minority_written / total_written if total_written > 0 else float("nan")

    print(f"\nOriginal training set: {total_rows:,} rows, failure rate {before_rate:.4%}")
    print(f"Resampled training set: {total_written:,} rows "
          f"({minority_written:,} minority + {majority_written:,} majority), failure rate {after_rate:.4%}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()