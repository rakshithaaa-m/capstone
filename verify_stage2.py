"""
Verify Stage 2 output: confirms row-count totals match Pass 1's scan, and
that every cleaned quarter file has a consistent schema (same columns,
same dtypes) -- both are things Stage 3 silently assumes are true.

Usage:
    python3 verify_stage2.py --dir data/processed/cleaned_by_quarter --expected_total 681960203
"""
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--expected_total", type=int, default=None,
                         help="Pass 1's reported total row count, to check nothing was lost/duplicated")
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("cleaned_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cleaned_*.parquet files found in {args.dir}")

    print(f"[CHECK] Found {len(files)} file(s)\n")

    total_rows = 0
    total_failures = 0
    schemas = {}
    problems = []

    for path in files:
        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows
        cols = tuple(pf.schema_arrow.names)
        dtypes = tuple(str(f.type) for f in pf.schema_arrow)

        total_rows += n_rows
        schemas[path.name] = (cols, dtypes)

        # Only read the failure column, not the whole file, to keep this fast/light
        fail_col = pd.read_parquet(path, columns=["failure"])
        n_fail = int(fail_col["failure"].sum())
        total_failures += n_fail

        print(f"  {path.name}: {n_rows:>12,} rows | {len(cols):>3} cols | {n_fail} failures")

    print(f"\n[TOTAL] {total_rows:,} rows across {len(files)} files | {total_failures:,} total failures")

    if args.expected_total is not None:
        diff = total_rows - args.expected_total
        if diff == 0:
            print(f"[OK] Matches Pass 1's reported total exactly ({args.expected_total:,})")
        else:
            print(f"[MISMATCH] Differs from Pass 1's reported total ({args.expected_total:,}) by {diff:,} rows")
            print("           Some rows may have been dropped/duplicated -- worth investigating which quarter.")

    # Schema consistency check
    first_file, (ref_cols, ref_dtypes) = next(iter(schemas.items()))
    mismatches = {name: (cols, dtypes) for name, (cols, dtypes) in schemas.items()
                  if cols != ref_cols or dtypes != ref_dtypes}

    if not mismatches:
        print(f"[OK] All {len(files)} files have identical schema ({len(ref_cols)} columns, matching dtypes)")
    else:
        print(f"[MISMATCH] {len(mismatches)} file(s) have a different schema than {first_file}:")
        for name, (cols, dtypes) in mismatches.items():
            print(f"  {name}:")
            missing = set(ref_cols) - set(cols)
            extra = set(cols) - set(ref_cols)
            if missing:
                print(f"    missing columns vs reference: {missing}")
            if extra:
                print(f"    extra columns vs reference: {extra}")
            common = set(ref_cols) & set(cols)
            dtype_diffs = {c: (ref_dtypes[ref_cols.index(c)], dtypes[cols.index(c)])
                           for c in common if ref_dtypes[ref_cols.index(c)] != dtypes[cols.index(c)]}
            if dtype_diffs:
                print(f"    dtype mismatches: {dtype_diffs}")


if __name__ == "__main__":
    main()
