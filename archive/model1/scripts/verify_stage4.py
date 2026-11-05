"""
Verify Stage 4 output: independently re-derives the checks the pipeline
already reported (row totals, date contiguity, failure rates) directly from
the actual parquet files, rather than trusting the printed run log.

Usage:
    python3 verify_stage4.py --dir data/processed --expected_total 681960203
"""
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def summarize(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    horizon_cols = [c for c in pf.schema_arrow.names if c.startswith("failure_within_")]
    horizon_col = horizon_cols[0] if horizon_cols else None

    cols_needed = ["date", "failure"] + (["is_censored"] if "is_censored" in pf.schema_arrow.names else [])
    if horizon_col:
        cols_needed.append(horizon_col)
    df = pd.read_parquet(path, columns=cols_needed)
    df["date"] = pd.to_datetime(df["date"])

    return {
        "rows": n_rows,
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "failure_rate": df["failure"].mean(),
        "censored_rate": df["is_censored"].mean() if "is_censored" in df.columns else float("nan"),
        "horizon_col": horizon_col,
        "horizon_rate": df[horizon_col].mean() if horizon_col else float("nan"),
        "horizon_positives": int(df[horizon_col].sum()) if horizon_col else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--expected_total", type=int, default=None,
                         help="Stage 3's reported total row count, to check nothing was lost/duplicated across the split")
    args = parser.parse_args()

    base = Path(args.dir)
    paths = {
        "train": base / "features_train.parquet",
        "val": base / "features_val.parquet",
        "test": base / "features_test.parquet",
    }
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split file(s): {missing} in {base}")

    summaries = {}
    for name, p in paths.items():
        print(f"[READING] {name}...")
        summaries[name] = summarize(p)

    print()
    total = 0
    for name, s in summaries.items():
        total += s["rows"]
        print(f"{name:5s}: {s['rows']:>12,} rows | {s['date_min'].date()} to {s['date_max'].date()}")
        print(f"       raw failure rate: {s['failure_rate']:.6%}  |  "
              f"{s['horizon_col'] or 'N-day target'} positive rate: {s['horizon_rate']:.6%} "
              f"({s['horizon_positives']:,} positives)  |  censored rate: {s['censored_rate']:.4%}")

    print(f"\n[TOTAL] {total:,} rows across train+val+test")
    if args.expected_total is not None:
        diff = total - args.expected_total
        if diff == 0:
            print(f"[OK] Matches expected total exactly ({args.expected_total:,})")
        else:
            print(f"[MISMATCH] Differs from expected total ({args.expected_total:,}) by {diff:,} rows")

    # Contiguity / non-overlap check
    if summaries["train"]["date_max"] >= summaries["val"]["date_min"]:
        print(f"[WARNING] train/val date ranges overlap or are out of order: "
              f"train max={summaries['train']['date_max'].date()}, val min={summaries['val']['date_min'].date()}")
    else:
        print(f"[OK] train ends ({summaries['train']['date_max'].date()}) before val begins "
              f"({summaries['val']['date_min'].date()})")

    if summaries["val"]["date_max"] >= summaries["test"]["date_min"]:
        print(f"[WARNING] val/test date ranges overlap or are out of order: "
              f"val max={summaries['val']['date_max'].date()}, test min={summaries['test']['date_min'].date()}")
    else:
        print(f"[OK] val ends ({summaries['val']['date_max'].date()}) before test begins "
              f"({summaries['test']['date_min'].date()})")

    if summaries["train"]["horizon_positives"] == 0:
        print("[WARNING] zero positive target examples in training -- model would have nothing to learn from.")
    if summaries["val"]["rows"] == 0 or summaries["test"]["rows"] == 0:
        print("[WARNING] val or test split is empty.")


if __name__ == "__main__":
    main()
