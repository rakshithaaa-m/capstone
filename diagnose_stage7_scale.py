"""
Diagnostic — run this BEFORE attempting Stage 7 training.

Reads ONLY the columns needed to answer "how many rows will Stage 7
actually need to train on" without ever loading the full 226-column,
484M-row file into memory. This tells us whether the failing-drives-only
filtering approach is actually going to bring things down to a tractable
size before we commit to rewriting and running the full training script.

Usage:
    python3 diagnose_stage7_scale.py \
        --train data/processed/features_train.parquet \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet
"""
import argparse
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    args = parser.parse_args()

    all_failing_serials = set()
    per_file_stats = {}

    for name, path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        pf = pq.ParquetFile(path)
        total_rows = pf.metadata.num_rows
        n_row_groups = pf.metadata.num_row_groups

        failing_serials_this_file = set()
        total_failures = 0
        # Stream through in row-group batches, reading ONLY the two columns
        # we need for this diagnostic — never the full 226-column row.
        for batch in pf.iter_batches(columns=["serial_number", "failure"], batch_size=500_000):
            tbl = batch.to_pandas()
            fails = tbl.loc[tbl["failure"] == 1, "serial_number"]
            failing_serials_this_file.update(fails.tolist())
            total_failures += len(fails)

        all_failing_serials.update(failing_serials_this_file)
        per_file_stats[name] = {
            "total_rows": total_rows,
            "row_groups": n_row_groups,
            "failure_rows": total_failures,
            "failing_serials": len(failing_serials_this_file),
        }
        print(f"[{name}] {total_rows:,} rows, {n_row_groups} row groups, "
              f"{total_failures} failure rows, {len(failing_serials_this_file)} distinct failing drives")

    print(f"\n[COMBINED] {len(all_failing_serials):,} distinct drives fail somewhere across train/val/test")
    print("This is the number of drives whose FULL history Stage 7 will actually need to load.")
    print("If this is in the thousands (not hundreds of millions), the failing-drives-only")
    print("filtering approach will make training tractable on your laptop.")


if __name__ == "__main__":
    main()
