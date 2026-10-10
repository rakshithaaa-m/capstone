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

The dashboard uses DuckDB for its serving layer. Anthropic is only needed
for live agent mode; simulation mode works without an API key. Set
`ANTHROPIC_MODEL` if the account should use a model other than the default.

```
pandas>=2.2
pyarrow>=15.0
numpy>=1.26
requests>=2.31
huggingface_hub>=0.23
duckdb>=1.0
joblib>=1.3
lightgbm>=4.3
scikit-learn>=1.4
optuna>=3.6
shap>=0.45
matplotlib>=3.8
torch>=2.2
streamlit>=1.49.0
anthropic>=0.40.0
```

## Dashboard serving layer

The prediction exports are large drive-day tables. Build the compact indexed
serving tables once after Stages 8-12, then start Streamlit:

```bash
python build_dashboard_db.py --results_dir results
streamlit run 13_dashboard.py -- --results_dir results
```

The builder creates one current row per physical drive (`drive_status`), a
small serial-to-model table, and an indexed mitigation lookup. The dashboard
uses those tables for fast selections and never loads the raw 145M-row CSVs
into the browser. By default the builder drops the copied large source tables
after creating the serving layer; use `--keep-source-tables` only if you want
them for offline rebuilding. Re-run with `--rebuild-sources` after regenerating
the prediction exports. If only the compact serving schema/query changed, use
`--refresh-serving`; it reuses native source tables when they were retained
with `--keep-source-tables` (otherwise it reads the CSVs directly).

## Full re-run after a model change (IMPORTANT)

Changing Stage 7 or Stage 8 **invalidates every downstream export**. The
RUL values feed the fusion, the fusion feeds the tier cutoffs, and the
cutoffs decide the tiers the dashboard and agent read. If you retrain the
LSTM or change the fusion encoding, you must re-run the whole chain below or
the dashboard will silently serve stale numbers.

Current state: the shipped `models/lstm_model.pt` and
`models/hybrid_config.json` on disk predate the Softplus-head and
horizon-bounded-fusion fixes, so `results/` is **stale** relative to the
code. Re-run in this order before regenerating any paper table:

```bash
# Stage 7 - LSTM (Softplus head; records full provenance in lstm_config.json)
python 07_train_lstm_v2.py \
  --train data/processed/features_train.parquet \
  --val   data/processed/features_val.parquet \
  --test  data/processed/features_test.parquet \
  --out_dir models results

# Stage 8 - fusion (meta-learner now uses the SAME horizon-bounded RUL
#           encoding as the weighted blend; see 08_hybrid_fusion.py)
python 08_hybrid_fusion.py <existing stage-8 args>

# Stage 9 - tier calibration (cutoffs must be re-derived from the new scores)
python 09_risk_tier_calibration.py <existing stage-9 args>

# Stage 10 - evaluation
python 10_evaluation.py <existing stage-10 args>

# Stage 11 - SHAP, then Stage 12 - mitigation mapping
python 11_shap_analysis.py <existing stage-11 args>
python 12_mitigation_mapping.py \
  --risk_tiers results/risk_tiers_test.csv \
  --shap_values results/shap_values_at_risk.parquet \
  --out results/mitigation_mapping.json

# Serving layer, then the paper tables
python build_dashboard_db.py --results_dir results
python 14b_paper_evaluation.py --results_dir results
```

Stage 7 is by far the longest step (it streams ~22 GB of training parquet);
budget hours, not minutes. Everything after it is minutes.

`models/hybrid_config.json` is the single source of truth for the fusion
formula, and `13_dashboard.py` / `14_agentic_mitigation.py` read it directly
— so after Stage 8 the dashboard's fusion labels update automatically with
no code change.

## Regenerating excluded artifacts

The following files are in `.gitignore` because they are large **and fully
derivable** from `data/processed/*.parquet`. A reviewer or teammate
reproduces them by re-running the stage that owns them. Nothing here is a
source of truth — the tracked `models/*.json`, `results/*_metrics.csv` and
`results/ablation_*.csv` are.

| Excluded artifact | Size | Regenerated by |
|---|---|---|
| `results/dashboard.duckdb` | ~160 MB | `python build_dashboard_db.py --results_dir results` |
| `results/hybrid_test_scores.csv` | ~13 GB | Stage 8 (`08_hybrid_fusion.py`) |
| `results/hybrid_val_scores.csv` | ~5 GB | Stage 8 (`08_hybrid_fusion.py`) |
| `results/risk_tiers_test.csv` | ~6.4 GB | Stage 9 (`09_risk_tier_calibration.py`) |
| `results/lgbm_test_probs.csv` | ~4.9 GB | Stage 6 (`06_train_lgbm.py`) |
| `results/lstm_full_test_preds.csv` | ~4.3 GB | Stage 7 (`07_train_lstm_v2.py`) |
| `results/shap_values_at_risk.parquet` | ~43 MB | Stage 11 (`11_shap_analysis.py`) |
| `results/shap_values.parquet` | ~3.5 MB | Stage 11 (`11_shap_analysis.py`) |
| `results/mitigation_mapping.json` | ~15 MB | Stage 12 (`12_mitigation_mapping.py`) |
| `models/lstm_model.pt` | ~130 KB | Stage 7 (`07_train_lstm_v2.py`) |

Stage 12 must be given the **full at-risk** SHAP file, not the small sample,
or mitigation coverage silently drops from 35,647 records to ~113:

```bash
python 12_mitigation_mapping.py \
  --risk_tiers results/risk_tiers_test.csv \
  --shap_values results/shap_values_at_risk.parquet \
  --out results/mitigation_mapping.json
```

After regenerating any of the prediction exports, rebuild the serving layer
so the dashboard fingerprint matches:

```bash
python build_dashboard_db.py --results_dir results
```

The dashboard tracks the size and mtime of six artifacts
(`risk_tiers_test.csv`, `hybrid_test_scores.csv`, `mitigation_mapping.json`,
`tier_cutoffs.json`, `lstm_metrics.csv`, `models/hybrid_config.json`). If any
of them change, every cached serving query is invalidated automatically and
the sidebar shows a "rebuild the serving DB" warning until you do.

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
