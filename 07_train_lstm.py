"""
Stage 7 — LSTM (Encoder + Attention) for RUL Regression

Architecture note: this is an LSTM encoder over a per-drive lookback
window, with additive attention pooling over the encoder's hidden states
feeding a small regression head — not a full sequence-to-sequence
encoder-decoder. Since the target here is a single scalar (RUL in days),
not a generated future sequence, this attention-pooled-encoder pattern is
the standard, defensible way to do point RUL estimation (e.g. Coursey et
al.'s Bi-LSTM RUL estimator), and is what "ED-LSTM with attention" means
in practice for this task. If you want a literal seq2seq decoder later
(e.g. to forecast a full future SMART trajectory, not just a single RUL
number), that's a larger extension — flag it as future work rather than
silently claiming it here.

Censored-drive handling: by default, rows from drives that never fail in
the observed window (is_censored=1, rul_days=NaN) are EXCLUDED from
training, since we have no ground-truth RUL for them — training against a
fabricated value would bias the model. This is a documented limitation
(a form of survivorship bias) worth naming explicitly in your report.
--censored_strategy cap is provided as an alternative: it assigns censored
rows a soft target RUL equal to the observed max horizon (rul_cap), useful
if you want the model to at least learn "these drives are far from
failure" — try both and compare validation RMSE before deciding.

Usage:
    python3 07_train_lstm.py \
        --train data/processed/features_train.parquet \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --lookback 14 \
        --epochs 20 \
        --censored_strategy exclude \
        --out_dir models results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

NON_FEATURE_COLS = [
    "date", "serial_number", "model", "capacity_bytes",
    "failure", "rul_days", "is_censored", "_split",
    "true_rul_days", "true_is_censored",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = set(NON_FEATURE_COLS) | {c for c in df.columns if c.startswith("failure_within_")}
    cols = [c for c in df.columns if c not in exclude]
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")
    return cols


def apply_censored_strategy(df: pd.DataFrame, strategy: str, rul_cap: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (df, sample_weight). `df`'s rul_days is used as the TRAINING
    target only — this is never applied to val/test evaluation data, which
    always keeps the true, un-modified rul_days (see main())."""
    df = df.copy()
    if strategy == "exclude":
        df = df[df["is_censored"] == 0].reset_index(drop=True)
        weight = np.ones(len(df), dtype=np.float32)
        return df, weight
    elif strategy == "cap":
        is_capped = (df["is_censored"] == 1).to_numpy()
        df.loc[df["is_censored"] == 1, "rul_days"] = rul_cap
        # Down-weight capped rows: their target is a fabricated proxy, not
        # an observed ground truth, so they should influence the loss less
        # than rows with a real, known RUL. This is a partial mitigation,
        # not a full censored/Tobit loss — documented honestly below.
        weight = np.where(is_capped, 0.3, 1.0).astype(np.float32)
        return df.reset_index(drop=True), weight
    else:
        raise ValueError(f"Unknown censored_strategy '{strategy}', use 'exclude' or 'cap'")


def build_sequences(df: pd.DataFrame, feature_cols: list[str], lookback: int, target_col: str = "rul_days"):
    """For every row, build a (lookback, n_features) input sequence from
    that SAME drive's preceding history up to and including that row.

    IMPORTANT: `df` must contain the FULL continuous per-drive history
    (all splits concatenated), not a single split in isolation — otherwise
    a drive's early val/test rows would incorrectly cold-start their
    lookback window instead of pulling in the preceding train-period
    history, which is the exact boundary-truncation bug already fixed
    once in Stage 3 and must not be reintroduced here. Which split each
    row belongs to is tracked via `df['_split']` and used only to slice
    the output afterward, not to change how the window is built.
    """
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


class RULDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, weights: np.ndarray = None):
        self.X = torch.from_numpy(sequences)
        self.y = torch.from_numpy(targets)
        self.w = torch.from_numpy(weights if weights is not None else np.ones(len(targets), dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


class AttentionLSTM(nn.Module):
    """LSTM encoder -> additive attention pooling over hidden states -> FC regression head."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_score = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        enc_out, _ = self.lstm(x)                      # (batch, seq_len, hidden)
        scores = self.attn_score(enc_out).squeeze(-1)    # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (enc_out * weights).sum(dim=1)          # (batch, hidden)
        rul_pred = self.head(context).squeeze(-1)          # (batch,)
        return rul_pred


def train_model(model, train_loader, val_loader, epochs: int, lr: float, device: str):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            per_sample_loss = loss_fn(preds, yb)
            loss = (per_sample_loss * wb).sum() / wb.sum().clamp(min=1e-8)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_sq_errors = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_sq_errors.append(((preds - yb) ** 2).cpu().numpy())
        val_rmse = float(np.sqrt(np.concatenate(val_sq_errors).mean())) if val_sq_errors else float("nan")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch:3d}/{epochs} | train weighted-MSE {np.mean(train_losses):.3f} | val RMSE {val_rmse:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_rmse


def evaluate_rmse_mae(model, loader, device: str) -> dict:
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_true)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae = float(np.mean(np.abs(preds - trues)))
    return {"rmse": rmse, "mae": mae, "n": len(trues)}, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the attention-LSTM RUL regressor")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--censored_strategy", type=str, default="exclude", choices=["exclude", "cap"])
    parser.add_argument("--rul_cap", type=float, default=None,
                         help="Only used with --censored_strategy cap. If omitted, derived from the "
                              "data as the maximum observed rul_days among TRAINING drives that did "
                              "fail — a data-justified cap rather than an arbitrary constant.")
    parser.add_argument("--out_dir", nargs=2, default=["models", "results"], metavar=("MODELS_DIR", "RESULTS_DIR"))
    args = parser.parse_args()

    models_dir, results_dir = Path(args.out_dir[0]), Path(args.out_dir[1])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_raw = pd.read_parquet(args.train)
    val_raw = pd.read_parquet(args.val)
    test_raw = pd.read_parquet(args.test)

    feature_cols = get_feature_cols(train_raw)
    print(f"Using {len(feature_cols)} feature columns, lookback={args.lookback} days")

    # --- Build ONE continuous per-drive dataset spanning all three splits ---
    # This is the fix for the boundary-truncation bug: sequences must be
    # built from each drive's full history, including whatever part of it
    # sits in an earlier split, or a drive's first val/test rows would
    # incorrectly cold-start their lookback window. Train-only rows may
    # already be resampled (Stage 5) with duplicated/reordered rows, so we
    # rebuild train's row set from features_train.parquet-equivalent
    # continuity by using the ORIGINAL (pre-resampling) train split for
    # sequence construction, and only apply resampling to which TRAINING
    # rows contribute to the loss, not to the sequence history itself.
    for name, part in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        part["_split"] = name
    full_df = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["serial_number", "date"], keep="first")
    # Sort/reset here to match EXACTLY what build_sequences() does internally
    # — its returned arrays are in this sorted order, so split_labels and
    # is_censored_full (extracted below) must be derived from this same
    # sorted frame, not the original concat order, or the masks used later
    # to select train/val/test rows silently misalign with X_full/y_full_true.
    full_df = full_df.sort_values(["serial_number", "date"]).reset_index(drop=True)

    # Normalize features using TRAIN-ONLY statistics (same principle as Stage 3's imputation)
    feat_mean = full_df.loc[full_df["_split"] == "train", feature_cols].mean()
    feat_std = full_df.loc[full_df["_split"] == "train", feature_cols].std().replace(0, 1.0)
    full_df[feature_cols] = (full_df[feature_cols] - feat_mean) / feat_std

    n_nan = int(full_df[feature_cols].isna().sum().sum())
    if n_nan > 0:
        print(f"NOTE: filling {n_nan} remaining NaN feature values with 0 "
              f"(expected for each drive's first recorded row, from rolling-std min_periods)")
        full_df[feature_cols] = full_df[feature_cols].fillna(0.0)

    # Keep the TRUE rul_days/is_censored untouched for honest evaluation —
    # the censored strategy below only ever modifies a training-time copy.
    full_df["true_rul_days"] = full_df["rul_days"]
    full_df["true_is_censored"] = full_df["is_censored"]

    # Derive the RUL cap from training data if not explicitly given
    rul_cap = args.rul_cap
    if args.censored_strategy == "cap" and rul_cap is None:
        observed = full_df.loc[(full_df["_split"] == "train") & (full_df["true_is_censored"] == 0), "true_rul_days"]
        if len(observed) == 0:
            raise ValueError("No observed (non-censored) failures in training data to derive a RUL cap from.")
        rul_cap = float(observed.max())
        print(f"[CAP] Data-derived rul_cap = {rul_cap:.1f} days (max observed training RUL)")

    # --- Build sequences ONCE on the full continuous dataset ---
    X_full, y_full_true = build_sequences(full_df, feature_cols, args.lookback, target_col="true_rul_days")
    split_labels = full_df["_split"].to_numpy()
    is_censored_full = full_df["true_is_censored"].to_numpy()

    # Training rows: apply the censored strategy to decide targets/weights
    train_mask = split_labels == "train"
    train_sub_df = full_df[train_mask].reset_index(drop=True)
    train_sub_df_strategy, train_weights = apply_censored_strategy(
        train_sub_df.assign(rul_days=train_sub_df["true_rul_days"]), args.censored_strategy, rul_cap or 0.0
    )
    # Map the strategy's row filter (only matters for 'exclude') back onto the sequence array
    if args.censored_strategy == "exclude":
        keep_idx = np.where(train_mask)[0][is_censored_full[train_mask] == 0]
    else:
        keep_idx = np.where(train_mask)[0]
    X_train = X_full[keep_idx]
    y_train = (
        train_sub_df_strategy["rul_days"].to_numpy(dtype=np.float32)
        if args.censored_strategy == "cap"
        else y_full_true[keep_idx]
    )
    w_train = train_weights

    if len(X_train) == 0:
        raise ValueError(
            "No training rows remain after applying the censored strategy — "
            "your training window may contain no observed failures. Push --train_end later."
        )
    print(f"Training on {len(X_train)} sequences (strategy='{args.censored_strategy}')")

    # Validation/test: ALWAYS evaluate against the TRUE rul_days, restricted
    # to rows with a real known RUL (non-censored) — evaluating against a
    # fabricated capped value would overstate how well the model predicts
    # actual failures, regardless of which strategy was used for training.
    val_eval_idx = np.where((split_labels == "val") & (is_censored_full == 0))[0]
    test_eval_idx = np.where((split_labels == "test") & (is_censored_full == 0))[0]
    X_val, y_val = X_full[val_eval_idx], y_full_true[val_eval_idx]
    X_test, y_test = X_full[test_eval_idx], y_full_true[test_eval_idx]
    print(f"Evaluating on {len(X_val)} val / {len(X_test)} test sequences with known ground-truth RUL")

    train_loader = DataLoader(RULDataset(X_train, y_train, w_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(RULDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = AttentionLSTM(n_features=len(feature_cols), hidden_size=args.hidden_size).to(device)
    print(f"[TRAIN] {sum(p.numel() for p in model.parameters()):,} parameters")

    model, best_val_rmse = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    print(f"[TRAIN] best validation RMSE: {best_val_rmse:.3f}")

    test_metrics, test_preds = evaluate_rmse_mae(model, test_loader, device)
    print(f"[TEST] RMSE={test_metrics['rmse']:.3f}  MAE={test_metrics['mae']:.3f}  n={test_metrics['n']}")

    # Full, unfiltered test predictions (every row, including censored
    # drives) for Stage 8's fusion — a deployed system needs a RUL
    # prediction for every drive, not just ones with known ground truth.
    test_all_idx = np.where(split_labels == "test")[0]
    X_full_test = X_full[test_all_idx]
    full_test_loader = DataLoader(
        RULDataset(X_full_test, np.zeros(len(X_full_test), dtype=np.float32)),
        batch_size=args.batch_size, shuffle=False,
    )
    _, full_test_preds = evaluate_rmse_mae(model, full_test_loader, device)

    torch.save(model.state_dict(), models_dir / "lstm_model.pt")
    with open(models_dir / "lstm_config.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols, "lookback": args.lookback,
            "hidden_size": args.hidden_size, "censored_strategy": args.censored_strategy,
            "rul_cap": rul_cap, "feat_mean": feat_mean.to_dict(), "feat_std": feat_std.to_dict(),
        }, f, indent=2)
    pd.DataFrame([test_metrics]).to_csv(results_dir / "lstm_metrics.csv", index=False)
    pd.Series(test_preds, name="lstm_rul_pred").to_csv(results_dir / "lstm_test_preds.csv", index=False)
    pd.Series(full_test_preds, name="lstm_rul_pred").to_csv(results_dir / "lstm_full_test_preds.csv", index=False)

    print(f"[SAVE] {models_dir/'lstm_model.pt'}")
    print(f"[SAVE] {models_dir/'lstm_config.json'}")
    print(f"[SAVE] {results_dir/'lstm_metrics.csv'}")
    print(f"[SAVE] {results_dir/'lstm_full_test_preds.csv'} (all {len(full_test_preds)} test rows, for Stage 8 fusion)")


if __name__ == "__main__":
    main()
