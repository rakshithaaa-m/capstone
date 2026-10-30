"""
Stage 7c -- Windowed RUL evaluation on the held-out test split.

Why this exists
---------------
TFILL (Ding et al., IEEE ISPA 2024) reports RMSE 4.97 / MAE 5.04 on a RUL
window CAPPED AT 60 DAYS before failure.  Our uncapped full-range MAE (96.0)
is a different task and is not comparable to that number.  This script
evaluates the ALREADY-TRAINED model under the base paper's exact window so
the two can be compared honestly.

It loads models/lstm_model.pt + models/lstm_config.json and reuses Stage 7's
own functions (imported, not copied) so the evaluation path is byte-identical
to a full retrain's -- only the training is skipped.

Nothing is overwritten: results are written to results/rul_window_metrics.csv.

Usage:
    python 14c_rul_window_eval.py --max_rul_for_eval 60
    python 14c_rul_window_eval.py --max_rul_for_eval 60 30 90 120 180
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).parent


def _load_stage7():
    """Import 07_train_lstm_v2.py as a module so we reuse its exact code."""
    spec = importlib.util.spec_from_file_location("stage7", HERE / "07_train_lstm_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="Windowed RUL evaluation on the test split")
    ap.add_argument("--train", default="data/processed/features_train.parquet")
    ap.add_argument("--val", default="data/processed/features_val.parquet")
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out", default="results/rul_window_metrics.csv")
    ap.add_argument("--max_history_days", type=int, default=180)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_rul_for_eval", type=float, nargs="+", default=[60.0])
    args = ap.parse_args()

    s7 = _load_stage7()
    models_dir = Path(args.models_dir)

    cfg = json.loads((models_dir / "lstm_config.json").read_text())
    feature_cols = list(cfg["feature_cols"])
    lookback = int(cfg["lookback"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = s7.AttentionLSTM(n_features=len(feature_cols),
                             hidden_size=int(cfg["hidden_size"])).to(device)
    model.load_state_dict(torch.load(models_dir / "lstm_model.pt", map_location=device))
    model.eval()
    print(f"[EVAL] loaded Stage 7 weights: {sum(p.numel() for p in model.parameters()):,} params, "
          f"head={cfg.get('head_activation', 'unknown')}, lookback={lookback}, device={device}")
    print(f"[EVAL] normalization restored from config "
          f"(best_val_rmse={cfg.get('best_val_rmse', float('nan')):.3f})")

    paths = {"train": args.train, "val": args.val, "test": args.test}
    print("[EVAL] identifying failing drives across train/val/test ...")
    failing_serials, last_date = s7.find_failing_serials(paths)

    test_raw = s7.load_failing_drives_only(args.test, failing_serials, feature_cols,
                                           last_date, args.max_history_days)
    print(f"[EVAL] test: {len(test_raw):,} rows loaded (failing drives only)")

    feat_mean = pd.Series(cfg["feat_mean"])
    feat_std = pd.Series(cfg["feat_std"]).replace(0, 1.0)
    test_raw[feature_cols] = (test_raw[feature_cols] - feat_mean) / feat_std
    n_nan = int(test_raw[feature_cols].isna().sum().sum())
    if n_nan:
        test_raw[feature_cols] = test_raw[feature_cols].fillna(0.0)
    test_arrays = s7.prepare_split_arrays(test_raw, feature_cols)
    del test_raw
    import gc
    gc.collect()

    base_idx = np.where(test_arrays["is_censored"] == 0)[0]
    print(f"[EVAL] {len(base_idx):,} sequences with known ground-truth RUL "
          f"(uncapped, the full-range task)")

    rows = []
    for cap in [None] + list(args.max_rul_for_eval):
        idx = base_idx
        if cap is not None:
            idx = idx[test_arrays["rul_days"][idx] <= cap]
        label = "full range" if cap is None else f"RUL <= {cap:g} days"
        if len(idx) == 0:
            print(f"[EVAL] {label}: 0 sequences, skipped")
            continue
        ds = s7.LazySequenceDataset(test_arrays["feat_arr"], test_arrays["drive_start_idx"],
                                    test_arrays["rul_days"], idx, lookback)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        m, _ = s7.evaluate_rmse_mae(model, loader, device)
        rows.append({"evaluation_window": label,
                     "max_rul_for_eval": cap if cap is not None else "full",
                     "n_sequences": int(m["n"]),
                     "rmse": float(m["rmse"]),
                     "mae": float(m["mae"])})
        print(f"[EVAL] {label:16s} n={int(m['n']):>9,}  RMSE {m['rmse']:8.3f}  MAE {m['mae']:8.3f}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[SAVE] {args.out}")

    print("\n=== COMPARISON WITH THE BASE PAPER (TFILL, Ding et al. ISPA 2024, Table IV) ===")
    print("  Base paper caps RUL at 60 days. Only the RUL <= 60 row is directly comparable.")
    print("  Base paper:  TFILL RMSE 4.97  MAE 5.04   (ST4000DM000, 2013-2023, single model)")
    for r in rows:
        if r["max_rul_for_eval"] != "full":
            print(f"  Ours:        RMSE {r['rmse']:.2f}  MAE {r['mae']:.2f}   "
                  f"({r['evaluation_window']}, n={r['n_sequences']:,}, full fleet)")
            print(f"  Ratio:       RMSE x{r['rmse'] / 4.97:.1f}   MAE x{r['mae'] / 5.04:.1f}")


if __name__ == "__main__":
    main()
