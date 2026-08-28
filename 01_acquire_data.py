"""
Stage 1 — Data Acquisition

Downloads raw quarterly Backblaze SMART stats ZIPs and extracts the CSVs
into data/raw/. Run this on the college lab machine (disk/RAM heavy, not
GPU-heavy) — it does not need a GPU.

Backblaze publishes quarterly ZIPs at:
  https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
Each quarter's page links to a ZIP such as "data_Q1_2024.zip" containing
one CSV per day. URLs change over time, so verify the exact link on the
page before running — pass it via --url or edit DEFAULT_QUARTERS below.

Usage:
    python3 01_acquire_data.py --url "https://.../data_Q1_2024.zip" --out data/raw
    python3 01_acquire_data.py --quarters 2024_Q1 2024_Q2   # uses DEFAULT_QUARTERS map
"""
import argparse
import sys
import zipfile
from pathlib import Path

import requests

# Fill in real URLs from the Backblaze downloads page before running.
# Example placeholders only — verify against the current page, links do change.
DEFAULT_QUARTERS = {
    "2024_Q1": "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2024.zip",
    "2024_Q2": "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q2_2024.zip",
}

CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    """Streams a large file to disk instead of loading it into memory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(f"\r  downloading {dest.name}: {pct:5.1f}%", end="", flush=True)
        print()
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extracts a ZIP and returns the list of CSV files found inside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            raise RuntimeError(f"Corrupt file in archive: {bad_file}")
        zf.extractall(out_dir)
    return sorted(out_dir.rglob("*.csv"))


def acquire_quarter(label: str, url: str, out_dir: Path) -> None:
    zip_path = out_dir / f"{label}.zip"
    print(f"[{label}] downloading from {url}")
    download_file(url, zip_path)
    print(f"[{label}] extracting...")
    csvs = extract_zip(zip_path, out_dir / label)
    print(f"[{label}] extracted {len(csvs)} CSV file(s) into {out_dir / label}")
    zip_path.unlink()  # remove the zip once extracted, saves disk space


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract Backblaze SMART data")
    parser.add_argument("--url", type=str, default=None,
                         help="Direct ZIP URL for a single quarter")
    parser.add_argument("--label", type=str, default="custom",
                         help="Label for the --url download (used as folder/file name)")
    parser.add_argument("--quarters", nargs="*", default=None,
                         help="Quarter labels to fetch from DEFAULT_QUARTERS, e.g. 2024_Q1 2024_Q2")
    parser.add_argument("--out", type=str, default="data/raw",
                         help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.url:
        acquire_quarter(args.label, args.url, out_dir)
    elif args.quarters:
        for q in args.quarters:
            if q not in DEFAULT_QUARTERS:
                print(f"Unknown quarter label '{q}'. Known: {list(DEFAULT_QUARTERS)}", file=sys.stderr)
                continue
            acquire_quarter(q, DEFAULT_QUARTERS[q], out_dir)
    else:
        parser.error("Provide either --url/--label for one file, or --quarters for known ones.")


if __name__ == "__main__":
    main()
