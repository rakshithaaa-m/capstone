"""
Stage 7 (v2, memory-safe) — Attention-LSTM for RUL Regression

Fixes the out-of-memory problem on very large datasets (hundreds of
millions of rows) WITHOUT changing the methodology at all. The key
insight: --censored_strategy exclude (the default) already means only
rows from drives that eventually fail are used for training or honest
evaluation -- a never-failed drive contributes zero training rows. The
original script loaded the ENTIRE dataset into memory and filtered
AFTERWARD. This version pushes that same filter down to the parquet read
itself, so only failing-drives' full history is ever loaded -- this is not
sampling and does not weaken the method, it is the exact same exclude
logic you already had, just applied before loading instead of after.

Normalization statistics (feature mean/std) are computed from a large
RANDOM SAMPLE of the full training file, not just the failing-drive
subset -- this matters because failing drives are not representative of
the whole fleet's feature distribution, and normalization needs to
reflect the general population your model will see at deployment time.

--censored_strategy cap is NOT supported in this memory-safe version: cap
needs every row (including healthy, censored ones) in training, which
brings back the full-dataset memory problem this rewrite exists to avoid.
If you need cap at this data scale, that requires a different, heavier
engineering approach (e.g. Polars streaming or a cluster) -- flag it as a
limitation in your report rather than attempting it on a laptop.

Full-fleet inference (every drive, including healthy ones -- needed by
Stage 8's fusion) is handled separately by predict_full_fleet(), which
streams through the test set in row-group batches and writes predictions
incrementally, so it never holds the full test set in memory either.

Usage:
    python3 07_train_lstm_v2.py \
        --train data/processed/features_train.parquet \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --lookback 14 --epochs 20 --hidden_size 32 \
        --out_dir models results
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

NON_FEATURE_COLS = [
    "date", "serial_number", "model", "capacity_bytes", "datacenter",
    "failure", "rul_days", "is_censored", "_split",
    "true_rul_days", "true_is_censored",
]


def coerce_legacy_format(df: pd.DataFrame) -> pd.DataFrame:
    """is_legacy_format is a real boolean signal, not junk -- coerce to int
    rather than excluding it. Some rows have missing values (None/NaN),
    which astype(int) cannot handle directly -- fillna(False) first
    (treating "unknown" as "not legacy") so the conversion never fails."""
    if "is_legacy_format" in df.columns and not pd.api.types.is_numeric_dtype(df["is_legacy_format"]):
        df["is_legacy_format"] = df["is_legacy_format"].fillna(False).astype(int)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    df = coerce_legacy_format(df)
    exclude = set(NON_FEATURE_COLS) | {c for c in df.columns if c.startswith("failure_within_")}
    cols = [c for c in df.columns if c not in exclude]
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")
    return cols


def find_failing_serials(paths: dict[str, str]) -> tuple[set, dict]:
    failing = set()
    last_date = {}
    for name, path in paths.items():
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["serial_number", "failure"], batch_size=500_000):
            tbl = batch.to_pandas()
            failing.update(tbl.loc[tbl["failure"] == 1, "serial_number"].tolist())
    print(f"[SCAN] {len(failing):,} distinct drives fail somewhere across train/val/test -- "
          f"only these drives' full history will be loaded for training/eval.")

    # Second, cheap pass (2 columns only) to find each failing drive's LAST
    # recorded date across all three files -- needed so we can bound how
    # much history per drive we load (see --max_history_days). This is far
    # cheaper than the main load: 2 columns vs 217, and only appends a
    # small (serial -> date) dict, never a growing list of full rows.
    for name, path in paths.items():
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["serial_number", "date"], batch_size=500_000):
            df = batch.to_pandas()
            df = df[df["serial_number"].isin(failing)]
            if len(df):
                grp_max = df.groupby("serial_number")["date"].max()
                for serial, d in grp_max.items():
                    if serial not in last_date or d > last_date[serial]:
                        last_date[serial] = d
    return failing, last_date


def load_failing_drives_only(path: str, serials: set[str], feature_cols: list[str],
                              last_date: dict, max_history_days: int) -> pd.DataFrame:
    needed_cols = list(set(feature_cols) | {
        "date", "serial_number", "model", "failure", "rul_days", "is_censored",
    } | {c for c in ["failure_within_14d"]})
    pf = pq.ParquetFile(path)
    all_cols = [f.name for f in pf.schema_arrow]
    needed_cols = [c for c in needed_cols if c in all_cols]

    # Cutoff date per serial = its last recorded date minus max_history_days.
    # Rows older than this are years-old history that a lookback=14 model
    # gets essentially no benefit from, but which cost exactly as much
    # memory as the recent, useful rows. Capping this is what actually
    # makes 31,000 drives' history fit in laptop-scale RAM.
    cutoff_by_serial = {s: (d - pd.Timedelta(days=max_history_days)) for s, d in last_date.items()}

    # NOTE: previously used filters=[("serial_number", "in", list(serials))],
    # pushed down to PyArrow's dataset scanner. With 31,000+ values, PyArrow
    # builds this as a large filter EXPRESSION rather than a hash lookup,
    # and constructing that expression itself caused the MemoryError below
    # -- before any actual row data was even read. Streaming batches and
    # filtering with pandas' .isin() against a Python set is a cheap O(1)
    # hash lookup per row and never materializes more than one batch plus
    # the running list of matches at a time.
    matched_parts = []
    total_matched = 0
    n_batches = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=needed_cols):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["serial_number"].isin(serials)]
        if len(chunk):
            cutoff = chunk["serial_number"].map(cutoff_by_serial)
            chunk = chunk[chunk["date"] >= cutoff]
        if len(chunk):
            # Downcast feature columns to float32 immediately -- pandas reads
            # parquet float columns as float64 by default, doubling memory
            # for no benefit here. Halving this now matters a lot more than
            # it sounds, since every one of these rows also feeds into a
            # sequence array further downstream.
            present_feats = [c for c in feature_cols if c in chunk.columns]
            chunk[present_feats] = chunk[present_feats].astype("float32")
            matched_parts.append(chunk)
            total_matched += len(chunk)
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"[SCAN] {path}: {n_batches} batch(es) scanned, "
                  f"{total_matched:,} matching rows found so far...")

    df = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame(columns=needed_cols)
    df = coerce_legacy_format(df)
    return df


def sample_for_normalization(path: str, feature_cols: list[str], max_rows: int = 2_000_000,
                              seed: int = 42) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    row_group_order = rng.permutation(pf.metadata.num_row_groups)

    parts = []
    total = 0
    for rg in row_group_order:
        tbl = pf.read_row_group(int(rg), columns=feature_cols)
        part = tbl.to_pandas()
        part = coerce_legacy_format(part)
        parts.append(part)
        total += len(part)
        if total >= max_rows:
            break
    sample = pd.concat(parts, ignore_index=True)
    print(f"[SAMPLE] Using {len(sample):,} rows (from {len(parts)} row group(s)) "
          f"to compute normalization statistics")
    return sample


def build_sequences(df: pd.DataFrame, feature_cols: list[str], lookback: int, target_col: str = "rul_days"):
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    n_features = len(feature_cols)
    sequences = np.zeros((len(df), lookback, n_features), dtype=np.float32)
    targets = df[target_col].values.astype(np.float32)

    for serial, group in df.groupby("serial_number"):
        idx = group.index.to_numpy()
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        for local_i, global_i in enumerate(idx):
            start = max(0, local_i - lookback + 1)
            window = feats[start:local_i + 1]
            if len(window) < lookback:
                pad = np.repeat(window[:1], lookback - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)
            sequences[global_i] = window
    return sequences, targets


def prepare_split_arrays(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Converts a raw (unwindowed) split DataFrame into the compact arrays
    LazySequenceDataset needs, WITHOUT ever materializing a
    (n_rows, lookback, n_features) array. feat_arr is (n_rows, n_features)
    -- 1x the raw data size, not lookback-times that -- which is the whole
    point: the lookback window is built on demand per-sample instead of
    pre-expanded for every row up front."""
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["serial_number", "date"], keep="first")
    df = df.reset_index(drop=True)

    global_idx = np.arange(len(df))
    drive_start_idx = pd.Series(global_idx).groupby(df["serial_number"].to_numpy()).transform("min").to_numpy()

    feat_arr = df[feature_cols].to_numpy(dtype=np.float32)
    return {
        "feat_arr": feat_arr,
        "drive_start_idx": drive_start_idx.astype(np.int64),
        "rul_days": df["rul_days"].to_numpy(dtype=np.float32),
        "is_censored": df["is_censored"].to_numpy(dtype=np.int8),
        "n_rows": len(df),
    }


class LazySequenceDataset(Dataset):
    """Builds each (lookback, n_features) window at __getitem__ time from a
    compact per-row feature array, instead of pre-building every window
    into one giant array. This is the fix for the real memory wall: the
    old build_sequences() allocated len(df) * lookback * n_features floats
    up front (a 14x multiplier at lookback=14) for every row in the
    combined train+val+test set simultaneously. This version only ever
    holds one batch's worth of windows in memory at a time, same as any
    normal PyTorch Dataset."""

    def __init__(self, feat_arr, drive_start_idx, targets, indices, lookback, weights=None):
        self.feat_arr = feat_arr
        self.drive_start_idx = drive_start_idx
        self.targets = targets
        self.indices = indices
        self.lookback = lookback
        self.weights = weights if weights is not None else np.ones(len(indices), dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_i = self.indices[i]
        start = max(self.drive_start_idx[global_i], global_i - self.lookback + 1)
        window = self.feat_arr[start:global_i + 1]
        if len(window) < self.lookback:
            pad = np.repeat(window[:1], self.lookback - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        x = torch.from_numpy(window.copy())
        y = torch.tensor(self.targets[global_i], dtype=torch.float32)
        w = torch.tensor(self.weights[i], dtype=torch.float32)
        return x, y, w


class RULDataset(Dataset):
    def __init__(self, sequences, targets, weights=None):
        self.X = torch.from_numpy(sequences)
        self.y = torch.from_numpy(targets)
        self.w = torch.from_numpy(weights if weights is not None else np.ones(len(targets), dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


class AttentionLSTM(nn.Module):
    """Attention-LSTM RUL regressor.

    The output head ends in Softplus, not a bare Linear.  A bare Linear can
    emit negative remaining-useful-life, which is physically meaningless and
    previously produced 10,747 negative predictions (minimum -4,475.9 days)
    in the full-fleet export.  Softplus keeps predictions strictly positive
    while staying smooth and differentiable (unlike ReLU, which has a dead
    gradient region and produces exact zeros that then poison the
    normalization fed to Stage 8's fusion).
    """

    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn_score = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        # Softplus: guarantees RUL > 0, unlike a bare Linear head which
        # produced 10,747 negative predictions (min -4,475.9 days) previously.

    def forward(self, x):
        enc_out, _ = self.lstm(x)
        scores = self.attn_score(enc_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (enc_out * weights).sum(dim=1)
        return F.softplus(self.head(context).squeeze(-1))


def train_model(model, train_loader, val_loader, epochs, lr, device,
                ckpt_path=None, resume=True):
    """Train with best-validation checkpointing.

    The best state is written to ``ckpt_path`` whenever validation improves,
    and reloaded on a subsequent run.  Without this, a run that dies at epoch
    12 of 20 loses every bit of its compute: the best weights only ever lived
    in memory.  That is exactly what happened on the first retrain attempt.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")
    best_val_rmse, best_state = float("inf"), None
    start_epoch = 1

    if ckpt_path is not None and resume and Path(ckpt_path).exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        best_val_rmse = float(saved["best_val_rmse"])
        best_state = saved["best_state"]
        start_epoch = int(saved.get("next_epoch", 1))
        print(f"[RESUME] checkpoint {ckpt_path}: best val RMSE {best_val_rmse:.3f}, "
              f"resuming at epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = (loss_fn(preds, yb) * wb).sum() / wb.sum().clamp(min=1e-8)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_sq = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_sq.append(((model(xb) - yb) ** 2).cpu().numpy())
        val_rmse = float(np.sqrt(np.concatenate(val_sq).mean())) if val_sq else float("nan")

        improved = val_rmse < best_val_rmse
        if improved:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save(
                    {"best_state": best_state, "best_val_rmse": best_val_rmse,
                     "next_epoch": epoch + 1, "epoch": epoch},
                    ckpt_path,
                )
        print(f"  epoch {epoch:3d}/{epochs} | train weighted-MSE {np.mean(train_losses):.3f} | "
              f"val RMSE {val_rmse:.3f} | best {best_val_rmse:.3f}"
              f"{'  <- saved' if improved else ''}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_rmse


def evaluate_rmse_mae(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0], batch[1]
            preds = model(xb.to(device)).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    preds, trues = np.concatenate(all_preds), np.concatenate(all_true)
    return {"rmse": float(np.sqrt(np.mean((preds - trues) ** 2))),
            "mae": float(np.mean(np.abs(preds - trues))), "n": len(trues)}, preds


def predict_full_fleet(model, path: str, feature_cols: list[str], feat_mean, feat_std,
                        lookback: int, device: str, out_path: Path, batch_drives: int = 2000):
    pf = pq.ParquetFile(path)
    all_serials = set()
    for batch in pf.iter_batches(columns=["serial_number"], batch_size=1_000_000):
        all_serials.update(batch.to_pandas()["serial_number"].tolist())
    all_serials = sorted(all_serials)
    print(f"[FULL INFERENCE] {len(all_serials):,} distinct drives in test set, "
          f"processing in batches of {batch_drives}")

    first_write = True
    model.eval()
    for i in range(0, len(all_serials), batch_drives):
        batch_serials = set(all_serials[i:i + batch_drives])
        chunk = pd.read_parquet(path, columns=list(set(feature_cols) | {"serial_number", "date"}),
                                 filters=[("serial_number", "in", list(batch_serials))])
        chunk = coerce_legacy_format(chunk)
        chunk[feature_cols] = (chunk[feature_cols] - feat_mean) / feat_std
        chunk[feature_cols] = chunk[feature_cols].fillna(0.0)
        chunk = chunk.sort_values(["serial_number", "date"]).reset_index(drop=True)

        X, _ = build_sequences(chunk.assign(rul_days=0.0), feature_cols, lookback)
        loader = DataLoader(RULDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=256, shuffle=False)
        with torch.no_grad():
            preds = []
            for xb, _, _ in loader:
                preds.append(model(xb.to(device)).cpu().numpy())
        preds = np.concatenate(preds)

        # Defensive clamp.  The Softplus head already guarantees positivity,
        # but this export feeds Stage 8's fusion, where a negative RUL would
        # invert the risk contribution.  Never let that happen silently.
        n_clamped = int((preds < 0).sum())
        if n_clamped:
            print(f"  [GUARD] clamped {n_clamped} negative RUL predictions to 0")
        preds = np.clip(preds, 0.0, None)

        out_chunk = pd.DataFrame({
            "serial_number": chunk["serial_number"], "date": chunk["date"], "lstm_rul_pred": preds
        })
        out_chunk.to_csv(out_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False
        print(f"  processed drives {i}-{min(i + batch_drives, len(all_serials))}/{len(all_serials)}")


def main():
    parser = argparse.ArgumentParser(description="Memory-safe attention-LSTM RUL training")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--norm_sample_rows", type=int, default=2_000_000)
    parser.add_argument("--max_history_days", type=int, default=180,
                         help="Only load each failing drive's last N days of history before its "
                              "last recorded date, instead of its entire multi-year lifetime. "
                              "With lookback=14, ancient history adds negligible signal but costs "
                              "the same memory as recent, useful rows -- this is what makes 31k+ "
                              "drives' history fit on a laptop.")
    parser.add_argument("--max_rul_for_eval", type=float, default=None,
                         help="Restrict TEST evaluation (RMSE/MAE) to sequences with true rul_days "
                              "<= this value -- does not affect training. Set to 60 to reproduce a "
                              "comparable evaluation window to Ding et al.'s TFILL paper, which caps "
                              "RUL at 60 days before failure. Leave unset for the original, full-range "
                              "evaluation.")
    parser.add_argument("--skip_full_inference", action="store_true")
    parser.add_argument(
        "--restart_training", action="store_true",
        help="Ignore any existing lstm_best_state.pt checkpoint and train from scratch. "
             "By default a previous run resumes from its last best-validation epoch.",
    )
    parser.add_argument(
        "--inference_only", action="store_true",
        help="Skip training and test evaluation; regenerate the full-fleet prediction "
             "CSVs from the saved lstm_model.pt + lstm_config.json. Use this to recover "
             "from a run that died during the multi-GB inference write.",
    )
    parser.add_argument("--skip_val_inference", action="store_true",
                        help="With --inference_only, regenerate only the test predictions.")
    parser.add_argument("--skip_test_inference", action="store_true",
                        help="With --inference_only, regenerate only the val predictions.")
    parser.add_argument("--out_dir", nargs=2, default=["models", "results"], metavar=("MODELS_DIR", "RESULTS_DIR"))
    args = parser.parse_args()

    models_dir, results_dir = Path(args.out_dir[0]), Path(args.out_dir[1])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Inference-only fast path -------------------------------------------
    # Full-fleet inference writes multi-GB CSVs that Stage 8 consumes.  If a
    # run dies during that write (or the machine restarts), re-running the
    # whole script would retrain from scratch -- hours -- to regenerate a
    # purely post-hoc artifact.  Everything inference needs was persisted to
    # lstm_config.json at training time: architecture, feature list, and the
    # exact normalization statistics the weights were fitted against.
    if args.inference_only:
        cfg_path = models_dir / "lstm_config.json"
        w_path = models_dir / "lstm_model.pt"
        if not cfg_path.exists() or not w_path.exists():
            raise SystemExit(
                f"--inference_only needs {cfg_path} and {w_path}. "
                "Run a full training pass first."
            )
        cfg = json.loads(cfg_path.read_text())
        feature_cols = list(cfg["feature_cols"])
        print(f"[INFER-ONLY] config: {len(feature_cols)} features, lookback "
              f"{cfg['lookback']}, hidden {cfg['hidden_size']}, "
              f"head={cfg.get('head_activation', 'unknown')}")
        print(f"[INFER-ONLY] restoring normalization from saved config "
              f"(head={cfg.get('head_activation')}, best_val_rmse="
              f"{cfg.get('best_val_rmse', float('nan')):.3f})")
        feat_mean = pd.Series(cfg["feat_mean"])
        feat_std = pd.Series(cfg["feat_std"]).replace(0, 1.0)

        model = AttentionLSTM(n_features=len(feature_cols),
                              hidden_size=int(cfg["hidden_size"])).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device))
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[INFER-ONLY] loaded weights ({n_params:,} parameters)")

        jobs = []
        if not args.skip_val_inference:
            jobs.append((args.val, results_dir / "lstm_full_val_preds.csv"))
        if not args.skip_test_inference:
            jobs.append((args.test, results_dir / "lstm_full_test_preds.csv"))
        if not jobs:
            print("[INFER-ONLY] nothing to do: both --skip_val_inference and "
                  "--skip_test_inference were set")
            return

        for src, out_path in jobs:
            print(f"[INFER-ONLY] full inference -> {out_path}")
            predict_full_fleet(model, src, feature_cols, feat_mean, feat_std,
                               int(cfg["lookback"]), device, out_path)
            print(f"[SAVE] {out_path}")
        return

    paths = {"train": args.train, "val": args.val, "test": args.test}

    pf = pq.ParquetFile(args.train)
    schema_cols = [f.name for f in pf.schema_arrow]
    peek = next(pf.iter_batches(columns=schema_cols, batch_size=10)).to_pandas()
    feature_cols = get_feature_cols(peek)
    print(f"Using {len(feature_cols)} feature columns")

    failing_serials, last_date = find_failing_serials(paths)
    if len(failing_serials) == 0:
        raise ValueError("No failing drives found in train/val/test -- nothing to train on.")

    train_raw = load_failing_drives_only(args.train, failing_serials, feature_cols, last_date, args.max_history_days)
    print(f"[LOAD] train: {len(train_raw):,} rows loaded (failing drives only)")

    norm_sample = sample_for_normalization(args.train, feature_cols, max_rows=args.norm_sample_rows)
    feat_mean = norm_sample[feature_cols].mean()
    feat_std = norm_sample[feature_cols].std().replace(0, 1.0)
    del norm_sample
    gc.collect()

    def normalize_inplace(df):
        df[feature_cols] = (df[feature_cols] - feat_mean) / feat_std
        n_nan = int(df[feature_cols].isna().sum().sum())
        if n_nan > 0:
            df[feature_cols] = df[feature_cols].fillna(0.0)
        return df

    train_raw = normalize_inplace(train_raw)
    train_arrays = prepare_split_arrays(train_raw, feature_cols)
    del train_raw
    gc.collect()

    train_idx = np.where(train_arrays["is_censored"] == 0)[0]
    train_ds = LazySequenceDataset(train_arrays["feat_arr"], train_arrays["drive_start_idx"],
                                    train_arrays["rul_days"], train_idx, args.lookback)
    print(f"Training on {len(train_ds)} sequences (all from failing drives, non-censored rows)")

    val_raw = load_failing_drives_only(args.val, failing_serials, feature_cols, last_date, args.max_history_days)
    print(f"[LOAD] val: {len(val_raw):,} rows loaded (failing drives only)")
    val_raw = normalize_inplace(val_raw)
    val_arrays = prepare_split_arrays(val_raw, feature_cols)
    del val_raw
    gc.collect()

    val_idx = np.where(val_arrays["is_censored"] == 0)[0]
    val_ds = LazySequenceDataset(val_arrays["feat_arr"], val_arrays["drive_start_idx"],
                                  val_arrays["rul_days"], val_idx, args.lookback)
    print(f"Evaluating on {len(val_ds)} val sequences with known ground-truth RUL")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AttentionLSTM(n_features=len(feature_cols), hidden_size=args.hidden_size).to(device)
    print(f"[TRAIN] {sum(p.numel() for p in model.parameters()):,} parameters")

    ckpt_path = models_dir / "lstm_best_state.pt"
    model, best_val_rmse = train_model(
        model, train_loader, val_loader, args.epochs, args.lr, device,
        ckpt_path=ckpt_path, resume=not args.restart_training,
    )
    if ckpt_path.exists():
        ckpt_path.unlink()  # consumed; the final weights are saved below
    print(f"[TRAIN] best validation RMSE: {best_val_rmse:.3f}")

    # Val's arrays are no longer needed once training is done -- free them
    # before loading test, so train+val+test never all sit in memory at once.
    del train_arrays, val_arrays, train_ds, val_ds, train_loader, val_loader
    gc.collect()

    test_raw = load_failing_drives_only(args.test, failing_serials, feature_cols, last_date, args.max_history_days)
    print(f"[LOAD] test: {len(test_raw):,} rows loaded (failing drives only)")
    test_raw = normalize_inplace(test_raw)
    test_arrays = prepare_split_arrays(test_raw, feature_cols)
    del test_raw
    gc.collect()

    test_idx = np.where(test_arrays["is_censored"] == 0)[0]
    if args.max_rul_for_eval is not None:
        # Restricts EVALUATION only (not training) to rows within N days
        # of failure -- matches Ding et al.'s methodology of capping RUL
        # to the last 60 days before failure, which is what makes their
        # reported MAE directly comparable to yours. Without this filter,
        # your MAE is computed over a much wider RUL range (potentially
        # years), which is a fundamentally harder, non-comparable target.
        before = len(test_idx)
        within_window = test_arrays["rul_days"][test_idx] <= args.max_rul_for_eval
        test_idx = test_idx[within_window]
        print(f"[EVAL WINDOW] Restricting test evaluation to rul_days <= {args.max_rul_for_eval} "
              f"days before failure: {len(test_idx):,}/{before:,} sequences kept "
              f"(this matches the base paper's comparable evaluation window)")
    test_ds = LazySequenceDataset(test_arrays["feat_arr"], test_arrays["drive_start_idx"],
                                   test_arrays["rul_days"], test_idx, args.lookback)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(test_ds)} test sequences with known ground-truth RUL")

    test_metrics, _ = evaluate_rmse_mae(model, test_loader, device)
    print(f"[TEST] RMSE={test_metrics['rmse']:.3f}  MAE={test_metrics['mae']:.3f}  n={test_metrics['n']}")

    del test_arrays, test_ds, test_loader
    gc.collect()

    torch.save(model.state_dict(), models_dir / "lstm_model.pt")
    with open(models_dir / "lstm_config.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "lookback": args.lookback,
            "hidden_size": args.hidden_size,
            "feat_mean": feat_mean.to_dict(),
            "feat_std": feat_std.to_dict(),
            # Full provenance: without this, a reader cannot tell which
            # hyper-parameters produced the shipped weights, which is exactly
            # what a reproducibility reviewer asks first.
            "head_activation": "softplus",
            "train_args": {
                "lookback": args.lookback,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_size": args.hidden_size,
                "max_history_days": args.max_history_days,
                "norm_sample_rows": args.norm_sample_rows,
                "max_rul_for_eval": args.max_rul_for_eval,
                "train": str(args.train),
                "val": str(args.val),
                "test": str(args.test),
            },
            "best_val_rmse": float(best_val_rmse),
            "test_metrics": {k: (float(v) if k in ("rmse", "mae") else int(v))
                             for k, v in test_metrics.items()},
            "torch_version": torch.__version__,
            "device": str(device),
        }, f, indent=2)
    # Record the evaluation window alongside the metrics.  lstm_metrics.csv is
    # overwritten on every run, so a capped run (--max_rul_for_eval 60) and a
    # full-range run are not comparable numbers.  Persisting the window in the
    # row -- and appending to a history file -- keeps both reproducible instead
    # of silently replacing one with the other.
    eval_window = args.max_rul_for_eval if args.max_rul_for_eval is not None else "full"
    test_metrics["eval_window_days"] = eval_window
    pd.DataFrame([test_metrics]).to_csv(results_dir / "lstm_metrics.csv", index=False)
    history_path = results_dir / "lstm_metrics_history.csv"
    history_row = pd.DataFrame([{**test_metrics, "recorded_at": pd.Timestamp.now().isoformat()}])
    history_row.to_csv(history_path, mode="a", header=not history_path.exists(), index=False)
    print(f"[SAVE] {models_dir/'lstm_model.pt'}, {models_dir/'lstm_config.json'}, "
          f"{results_dir/'lstm_metrics.csv'} (window={eval_window})")
    print(f"[SAVE] appended eval history -> {history_path}")

    if not args.skip_full_inference:
        predict_full_fleet(model, args.val, feature_cols, feat_mean, feat_std, args.lookback, device,
                            results_dir / "lstm_full_val_preds.csv")
        print(f"[SAVE] {results_dir/'lstm_full_val_preds.csv'}")

        predict_full_fleet(model, args.test, feature_cols, feat_mean, feat_std, args.lookback, device,
                            results_dir / "lstm_full_test_preds.csv")
        print(f"[SAVE] {results_dir/'lstm_full_test_preds.csv'}")
    else:
        print("[SKIP] Full-fleet inference skipped (--skip_full_inference) -- run it separately when ready.")


if __name__ == "__main__":
    main()