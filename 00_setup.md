# Stage 0 — Environment Setup (Laptop + External HDD)

Your college lab machine isn't available, so the full pipeline — ETL and
model training — runs on your own laptop, with an external HDD holding
the raw/processed data your laptop's internal storage can't fit. This
changes a few practical things from earlier guidance:

- **HDDs are slow at random access**, not just sequential reads. Stages 2-3
  (dedup, groupby, rolling windows) do scattered, non-sequential I/O — this
  will be noticeably slower on HDD than it would be on an SSD or the lab
  machine. Budget more time per run, and avoid rerunning stages
  unnecessarily while debugging (check printed diagnostics before moving
  to the next stage, as established throughout this pipeline).
- **Start with ONE quarter**, not your full historical range — this matters
  even more now, since a bug discovered after processing months of data on
  a slow drive costs you real wall-clock time, not just disk space.
- Point `data/raw/` and `data/processed/` at your external HDD's mount
  path (e.g. `E:\hdd-project\data` on Windows or `/Volumes/YourDrive/hdd-project/data`
  on macOS), not your laptop's internal drive, via the `--raw_dir`/`--out`/
  `--out_dir` arguments each script already exposes.

## Environment setup (on your laptop)

```bash
# create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# confirm GPU is visible (only relevant once you reach Stage 7's LSTM training)
python -c "import torch; print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False` on a laptop that does have a
discrete GPU, that's a driver/CUDA-toolkit mismatch worth fixing before
Stage 7 — Stages 0-5 here don't need a GPU at all, so it's not blocking
for this part of the pipeline.

## requirements.txt

```
pandas>=2.2
pyarrow>=15.0
numpy>=1.26
requests>=2.31
huggingface_hub>=0.23
lightgbm>=4.3
scikit-learn>=1.4
optuna>=3.6
shap>=0.45
matplotlib>=3.8
torch>=2.2
streamlit>=1.33
```

## Repo layout to create

```
project/
  data/                                # point this at your external HDD
    raw/            # Stage 1 output (gitignored, too large for git)
    processed/      # Stage 2-5 outputs (parquet, push to HF Hub instead of git)
  scripts/
    01_acquire_data.py
    02_clean_data.py
    03_build_targets_and_features.py   # runs on the FULL dataset, before splitting
    04_temporal_split.py               # runs LAST, splits the already-featured data
    05_imbalance_strategy.py
  models/
  results/
  requirements.txt
  .gitignore          # add data/ and models/*.pkl/.pt — push those to HF Hub, not git
```

## Push processed data to Hugging Face Hub (do this at the end of Stage 4)

Doing this isn't just good practice now — it's your backup. An external
HDD is a single point of failure for a month-long project; treat the HF
Hub push as insurance, not an optional extra.

```bash
huggingface-cli login   # one-time, paste your HF access token
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='data/processed/features_train.parquet',
    path_in_repo='features_train.parquet',
    repo_id='YOUR_USERNAME/hdd-failure-features',
    repo_type='dataset'
)
"
```

## If you still get ARC Labs GPU access for Stage 7 specifically

Nothing about Stages 0-5 changes — they were always CPU/RAM-bound, not
GPU-bound. If ARC Labs (or Kaggle, as a free fallback) becomes available
later for the LSTM training in Stage 7, you'd just push your Stage 4
output to Hugging Face Hub as above and pull it down there — the pipeline
was already designed around that handoff.
