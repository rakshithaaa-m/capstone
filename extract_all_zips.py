"""
Bulk-extracts all Backblaze quarterly ZIPs sitting in one staging folder
into the data/raw/<quarter_label>/*.csv layout that 02_clean_data.py's
discover_chunks() expects.

Place all downloaded ZIPs in one folder first (e.g. data/raw_zips/), then:

    python extract_all_zips.py --zips_dir data/raw_zips --out_dir data/raw

Quarter label is derived from the ZIP filename. Handles two common Backblaze
naming patterns:
    data_Q1_2024.zip           -> label "2024_Q1"
    2024Q1.zip                 -> label "2024_Q1"
If a filename doesn't match either pattern, the label falls back to the
filename stem as-is (e.g. "custom_batch.zip" -> "custom_batch") so nothing
is silently skipped -- check the printed label if that happens and rename
the source zip if you want a cleaner folder name.

CSVs found nested inside subfolders within a ZIP (Backblaze ZIPs sometimes
extract into an internal folder) are flattened into data/raw/<label>/ directly,
since discover_chunks() does an rglob() so nesting doesn't break Stage 2,
but keeping it flat avoids confusion when browsing the folder later.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

QUARTER_PATTERNS = [
    re.compile(r"[Qq]([1-4])_?(\d{4})"),   # Q1_2024 or Q12024
    re.compile(r"(\d{4})_?[Qq]([1-4])"),   # 2024_Q1 or 2024Q1
]


def derive_label(zip_name: str) -> str:
    stem = Path(zip_name).stem
    for pat in QUARTER_PATTERNS:
        m = pat.search(stem)
        if m:
            g1, g2 = m.groups()
            # figure out which group is the year (4 digits) vs quarter (1 digit)
            year, q = (g1, g2) if len(g1) == 4 else (g2, g1)
            return f"{year}_Q{q}"
    return stem  # fallback: use the filename as-is


def extract_one(zip_path: Path, out_root: Path) -> tuple[str, int]:
    label = derive_label(zip_path.name)
    target_dir = out_root / label
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt file inside {zip_path.name}: {bad}")

        # Extract to a temp subfolder first, then flatten any nested CSVs
        # up to target_dir directly -- avoids collisions if two zips
        # happen to contain identically-named internal subfolders.
        tmp_extract = target_dir / "_tmp_extract"
        tmp_extract.mkdir(exist_ok=True)
        zf.extractall(tmp_extract)

    moved = 0
    for csv_path in tmp_extract.rglob("*.csv"):
        dest = target_dir / csv_path.name
        if dest.exists():
            print(f"    WARNING: {dest.name} already exists in {label}, skipping duplicate from {zip_path.name}")
            continue
        shutil.move(str(csv_path), str(dest))
        moved += 1

    shutil.rmtree(tmp_extract)
    return label, moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-extract Backblaze quarterly ZIPs")
    parser.add_argument("--zips_dir", type=str, required=True, help="Folder containing all downloaded ZIP files")
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Destination, matches Stage 2's --raw_dir")
    parser.add_argument("--delete_zips", action="store_true",
                         help="Delete each ZIP after successful extraction, to save external-HDD space")
    args = parser.parse_args()

    zips_dir = Path(args.zips_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(zips_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {zips_dir}")

    print(f"[FOUND] {len(zip_files)} zip file(s) in {zips_dir}")

    total_csvs = 0
    for zp in zip_files:
        print(f"[EXTRACT] {zp.name} ...")
        label, n_csvs = extract_one(zp, out_dir)
        total_csvs += n_csvs
        print(f"    -> {out_dir / label} ({n_csvs} CSV file(s))")
        if args.delete_zips:
            zp.unlink()
            print(f"    (deleted {zp.name} to free space)")

    print(f"\n[DONE] Extracted {len(zip_files)} zip(s), {total_csvs} total CSV files, into {out_dir}")
    print(f"       Quarter folders created: {sorted(p.name for p in out_dir.iterdir() if p.is_dir())}")


if __name__ == "__main__":
    main()
