"""
Stage 6 — LightGBM Classifier (TWO-MODE, PROCESS-ISOLATED)

CHANGE IN THIS VERSION vs. the last one you ran:

1. Added a bitness check printed at startup. Run `--mode predict` once and
   look at the very first line of output. If it says "32-bit interpreter"
   STOP and fix that first — nothing below will reliably help if so, since
   a 32-bit process is capped at ~2-4GB of address space regardless of how
   much physical RAM is free. That symptom (small, routine allocations
   failing at a consistent row count, unaffected by chunk size or dtype
   changes) is exactly what a 32-bit process looks like from the outside.

2. stream_predict_booster no longer builds growing Python lists of every
   chunk's probs/labels and concatenates them at the end. That pattern
   holds ALL completed chunks in memory simultaneously with whatever chunk
   is currently mid-flight, and then needs a second full-size allocation
   for the final concatenate — real, avoidable peak-memory overhead. This
   version pre-allocates one fixed-size float32/int8 output array up front
   (size known from parquet metadata) and fills it slice-by-slice instead.
   No growing list, no final concatenate, ~4x smaller footprint per row
   than the old float64 probs + int64 labels lists.

   This alone will not fix a genuine 32-bit address-space cap — check (1)
   first. But it removes a real source of unnecessary memory pressure
   either way, and is simpler than the resumable/checkpointing version
   from the last attempt, which added complexity without addressing either
   of the two things above.

Usage unchanged — two separate command invocations:

    python 06_train_lightgbm.py --mode train \
        --train data/processed/train_resampled.parquet \
        --val data/processed/features_val.parquet \
        --target failure_within_14d \
        --n_trials 30 \
        --out_dir models results

    python 06_train_lightgbm.py --mode predict \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --cost_matrix data/processed/cost_matrix.json \
        --target failure_within_14d \
        --out_dir models results
"""
import argparse
import gc
import json
import struct
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import psutil
    _PROCESS = psutil.Process()
except ImportError:
    _PROCESS = None
    print("[ENV] psutil not installed -- RSS memory logging disabled. "
          "Run 'pip install psutil --break-system-packages' (or just 'pip install psutil' "
          "on Windows) to enable it -- it's the fastest way to see whether memory is "
          "actually growing chunk over chunk, instead of guessing.")
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

NON_FEATURE_COLS = [
    "date", "serial_number", "model", "capacity_bytes",
    "failure", "rul_days", "is_censored", "datacenter",
]
ROW_CHUNK_SIZE = 500_000
PREDICT_CHUNK_SIZE = 200_000

# Infrastructure / deployment identifiers.  These are NOT SMART health
# signals -- they encode which datacenter, pod, or slot a drive occupies.
# SHAP ranked vault_id as the single strongest driver (mean |SHAP| 0.293,
# ahead of error_total at 0.211), and fix1_vault_id_ablation.py measured that
# REMOVING them raises test AUPRC by 3.3%.  A model whose top feature is
# datacenter identity will not transfer to an unseen site, so they are
# excluded by default.  Override with --keep_infra_features if a reviewer
# asks for the original configuration.
INFRA_FEATURE_COLS = [
    "vault_id",
    "pod_id",
    "cluster_id",
    "pod_slot_num",
    "is_legacy_format",
]
VAL_TUNING_SAMPLE_ROWS = 5_000_000  # fixed sample used ONLY for Optuna's
                                     # repeated trials, never for any final
                                     # reported number.


def print_environment_diagnostics() -> None:
    bits = struct.calcsize("P") * 8
    print(f"[ENV] Python interpreter: {bits}-bit ({sys.executable})")
    if bits == 32:
        print("[ENV] *** WARNING: 32-bit interpreter detected. ***")
        print("[ENV] This caps process address space at ~2-4GB regardless of free")
        print("[ENV] system RAM, and produces exactly the symptom you've been seeing:")
        print("[ENV] small, routine allocations failing at a consistent row count.")
        print("[ENV] Fix: rebuild this venv from a 64-bit Python before continuing.")
    try:
        import lightgbm
        print(f"[ENV] lightgbm version: {lightgbm.__version__}")
    except Exception:
        pass
    print(f"[ENV] pyarrow version: {pa.__version__}")


def log_memory(tag: str) -> None:
    """Prints real process RSS (if psutil is available) and PyArrow's
    internal memory pool usage. PyArrow keeps its own C++ allocator pool
    and does not always release freed memory back to the OS immediately --
    if 'pyarrow pool' climbs steadily across chunks while row count grows,
    that -- not Python-level fragmentation -- is very likely the real
    cause of allocation failures at a consistent row count regardless of
    chunk size."""
    parts = [tag]
    if _PROCESS is not None:
        rss_mb = _PROCESS.memory_info().rss / 1024**2
        parts.append(f"RSS={rss_mb:,.0f}MB")
    pool_mb = pa.total_allocated_bytes() / 1024**2
    parts.append(f"pyarrow_pool={pool_mb:,.0f}MB")
    print("[MEM] " + " | ".join(parts))


def coerce_legacy_format(df: pd.DataFrame) -> pd.DataFrame:
    """is_legacy_format is a real boolean signal, not junk -- coerce to int
    rather than excluding it. fillna(False) first so astype never fails on
    a null (treats "unknown" as "not legacy", the more common case)."""
    if "is_legacy_format" in df.columns and not pd.api.types.is_numeric_dtype(df["is_legacy_format"]):
        df["is_legacy_format"] = df["is_legacy_format"].fillna(False).astype(int)
    return df


def get_feature_cols(df: pd.DataFrame, target: str,
                     extra_exclude: list[str] | None = None) -> list[str]:
    exclude = set(NON_FEATURE_COLS) | {target}
    exclude |= {c for c in df.columns if c.startswith("failure_within_") and c != target}
    if extra_exclude:
        present = [c for c in extra_exclude if c in df.columns]
        missing = [c for c in extra_exclude if c not in df.columns]
        if missing:
            print(f"[FEATURES] requested exclusions not present in the data, ignoring: {missing}")
        if present:
            print(f"[FEATURES] Excluding {len(present)} infrastructure identifier(s): {present}")
        exclude |= set(present)
    cols = [c for c in df.columns if c not in exclude]

    df = coerce_legacy_format(df)

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found, exclude or encode them: {non_numeric}")
    return cols


def downcast_floats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    float_cols = [c for c in cols if pd.api.types.is_float_dtype(df[c])]
    if float_cols:
        df[float_cols] = df[float_cols].astype("float32")
    return df


def load_train_chunked(path: Path, target: str,
                       extra_exclude: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    parquet_file = pq.ParquetFile(path)
    feature_cols = None
    chunks = []
    rows_so_far = 0
    for batch in parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE):
        df = batch.to_pandas()
        if feature_cols is None:
            feature_cols = get_feature_cols(df, target, extra_exclude)
        else:
            df = coerce_legacy_format(df)
        df = downcast_floats(df, feature_cols)
        chunks.append(df[feature_cols + [target]])
        rows_so_far += len(df)
        if rows_so_far % (ROW_CHUNK_SIZE * 10) == 0:
            print(f"[LOAD] {rows_so_far:,} training rows read so far...")
    print(f"[LOAD] Concatenating {rows_so_far:,} already-downcasted rows...")
    full = pd.concat(chunks, ignore_index=True)
    del chunks
    return full, feature_cols


def load_val_tuning_sample(path: Path, feature_cols: list[str], target: str,
                            max_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    parquet_file = pq.ParquetFile(path)
    chunks = []
    rows_so_far = 0
    for batch in parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE, columns=feature_cols + [target]):
        df = batch.to_pandas()
        df = coerce_legacy_format(df)
        df = downcast_floats(df, feature_cols)
        chunks.append(df)
        rows_so_far += len(df)
        if rows_so_far >= max_rows:
            break
    sample = pd.concat(chunks, ignore_index=True)
    print(f"[VAL SAMPLE] Using {len(sample):,} rows (of the full validation set) for Optuna tuning only")
    return sample[feature_cols], sample[target]


def stream_predict_booster(booster: lgb.Booster, path: Path, feature_cols: list[str],
                            target: str, want_keys: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Pre-allocates the full output arrays up front (size known from parquet
    metadata) instead of accumulating a growing list of per-chunk arrays and
    concatenating at the end. This avoids holding every completed chunk's
    result in memory simultaneously with the chunk currently being computed,
    and avoids a second full-size allocation for a final concatenate.
    Probs stored as float32 and labels as int8 -- both are exact for this
    use case (probabilities and binary 0/1 labels) and use a quarter to an
    eighth of the memory of the previous float64/int64 lists.
    """
    total_rows = pq.ParquetFile(path).metadata.num_rows
    out_probs = np.empty(total_rows, dtype="float32")
    out_labels = np.empty(total_rows, dtype="int8")
    # Join keys are stored as fixed-width byte arrays, not object arrays.
    # A 145M-row object array holds 145M PyObject pointers PLUS the interned
    # string objects behind them -- several GB apiece, and two of them is
    # what pushed this job past the machine's available RAM on the 5-booster
    # ensemble pass.  "<U20" is 20 bytes/row (160 MB at 145M) and byte order
    # is irrelevant for an opaque join key; dates become "<U10" (YYYY-MM-DD).
    # Join keys are byte-identical on every ensemble pass over the same file, so
    # passes 2..N pass want_keys=False and reuse pass 1's arrays.  Holding two
    # copies (the retained set plus the in-flight set) is ~9 GB at 145M rows and
    # drove this machine into paging on 2026-10-23, cutting throughput from
    # ~20 chunks/min to below 1.
    out_serials = np.empty(total_rows, dtype="S20") if want_keys else None
    out_dates = np.empty(total_rows, dtype="S10") if want_keys else None

    read_cols = feature_cols + [target, "serial_number", "date"]
    parquet_file = pq.ParquetFile(path)
    pos = 0
    n_chunks = 0
    log_memory("baseline before streaming")
    for batch in parquet_file.iter_batches(batch_size=PREDICT_CHUNK_SIZE, columns=read_cols):
        df = batch.to_pandas()
        df = coerce_legacy_format(df)
        X = df[feature_cols].astype("float32")
        probs = booster.predict(X)
        n = len(df)

        out_probs[pos:pos + n] = probs
        out_labels[pos:pos + n] = df[target].to_numpy()
        if want_keys:
            out_serials[pos:pos + n] = df["serial_number"].astype("S20").to_numpy()
            out_dates[pos:pos + n] = df["date"].astype("S10").to_numpy()
        pos += n

        del df, X, probs
        n_chunks += 1

        # Arrow's default pool retains freed blocks for reuse. Over a 145M-row
        # stream it grows without bound (~10 MB/chunk observed), which drove the
        # machine into paging and cut throughput from ~25 to <1 chunk/min on
        # 2026-10-23. Releasing every 10 chunks keeps the pool bounded without
        # paying the allocator cost on every chunk.
        if n_chunks % 10 == 0:
            pa.default_memory_pool().release_unused()

        if n_chunks % 20 == 0:
            print(f"[STREAM] {n_chunks} chunk(s) / {pos:,} rows processed so far...")
            log_memory(f"chunk {n_chunks}")
            gc.collect()
            pa.default_memory_pool().release_unused()

    # pos should equal total_rows, but slice defensively in case parquet
    # metadata and actual row count ever disagree
    return (out_labels[:pos], out_probs[:pos],
            out_serials[:pos] if want_keys else None,
            out_dates[:pos] if want_keys else None)


def decode_keys(arr: np.ndarray) -> np.ndarray:
    """Turn a fixed-width byte array back into a pandas string Series.

    The streaming path stores join keys as '<U20'/'<U10' byte arrays to keep
    peak memory flat on a 145M-row split.  Pandas writes np.bytes_ as
    ``b'ZA1C0JYE'`` into CSV, which would silently corrupt every downstream
    join, so decode once at write time.
    """
    return pd.Series(arr).str.decode("utf-8", errors="replace")


def ck_sig_of(boosters: list) -> str:
    """Stable identity for an ensemble, so a checkpoint built from one seed
    list is never reused for a different one."""
    parts = []
    for b in boosters:
        f = getattr(b, "model_file", None) or getattr(b, "name", None) or repr(b)
        parts.append(Path(str(f)).name if isinstance(f, str) else str(f))
    return ",".join(parts)


def stream_predict_ensemble(boosters: list, path: Path, feature_cols: list[str],
                            target: str, ckpt_dir: Path | None = None):
    """Average the probability output of several boosters over one split.

    Re-uses the single-model streaming path once per booster and averages the
    result.  Memory cost is one extra full-size float32 array (the accumulated
    average), not N copies of the chunk data, and the parquet is re-read per
    booster -- slower, but it keeps peak RSS identical to the single-model
    case, which matters on a 16 GB laptop.

    If ``ckpt_dir`` is given, the running sum is written there after every pass
    together with the join keys, labels and pass count.  A 5-pass ensemble over
    145M rows can run for many hours, and an OOM kill near the end otherwise
    throws all of it away.  A completed checkpoint is reused on the next run,
    so a retry resumes at the pass after the last one recorded.
    """
    n_models = len(boosters)
    if n_models == 1 and ckpt_dir is None:
        return stream_predict_booster(boosters[0], path, feature_cols, target)

    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    acc = labels = serials = dates = None
    start_pass = 1
    if ckpt_dir is not None:
        done = ckpt_dir / "acc.npy"
        meta_p = ckpt_dir / "meta.npz"
        if done.exists() and meta_p.exists():
            try:
                m = np.load(meta_p, allow_pickle=False)
                ck_done = int(m["passes_done"])
                ck_total = int(m["n_models_total"])
                ck_sig = str(m["signature"])
                # A checkpoint is only valid for the SAME ensemble it was built
                # from.  Reusing a sum from a different seed list would divide a
                # 2-model partial sum by 3 and silently produce wrong
                # probabilities, so verify the identity before trusting it.
                sig = f"{n_models}|{getattr(ckpt_dir, '_sig', '')}"
                if ck_total == n_models and ck_sig == ck_sig_of(boosters) and n_models > 1:
                    acc = np.load(done)
                    labels, serials, dates = m["labels"], m["serials"], m["dates"]
                    start_pass = ck_done + 1
                    print(f"[RESUME] checkpoint valid: {ck_done}/{n_models} passes already "
                          f"accumulated, resuming at pass {start_pass}")
                else:
                    print(f"[RESUME] checkpoint IGNORED: it was built for {ck_total} boosters "
                          f"(signature {ck_sig}) but this run has {n_models} "
                          f"(signature {ck_sig_of(boosters)}). Restarting from pass 1.")
            except Exception as exc:
                print(f"[RESUME] checkpoint unreadable ({exc}); restarting from pass 1")
                acc = labels = serials = dates = None
                start_pass = 1

    for i in range(start_pass, n_models + 1):
        print(f"[ENSEMBLE]   pass {i}/{n_models}")
        # Only the first pass captures join keys; later passes reuse them.  On a
        # resume the keys come from the checkpoint, so serials is already set.
        want_keys = serials is None
        labels_i, probs, serials_i, dates_i = stream_predict_booster(
            boosters[i - 1], path, feature_cols, target, want_keys=want_keys
        )
        if want_keys:
            labels, serials, dates = labels_i, serials_i, dates_i
        del labels_i, serials_i, dates_i
        # float32 accumulation: 5 summed probabilities stay well inside
        # float32 precision, and it halves the accumulator (0.58 GB vs 1.16 GB
        # at 145M rows) which matters when the process is already near the
        # machine's memory ceiling.
        acc = probs.copy() if acc is None else acc + probs
        del probs
        gc.collect()
        if ckpt_dir is not None:
            np.save(ckpt_dir / "acc.npy", acc)
            np.savez(ckpt_dir / "meta.npz", labels=labels, serials=serials,
                     dates=dates, passes_done=np.int64(i),
                     n_models_total=np.int64(n_models),
                     signature=np.array(ck_sig_of(boosters)))
            print(f"[CHECKPOINT] pass {i}/{n_models} saved to {ckpt_dir}")

    return labels, (acc / n_models).astype(np.float32), serials, dates


def tune_with_optuna(X_train, y_train, X_val_sample, y_val_sample, n_trials: int, seed: int = 42) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "random_state": seed,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val_sample)[:, 1]
        if y_val_sample.sum() == 0:
            raise ValueError(
                "Validation sample has zero positive examples -- cannot tune AUPRC. "
                "Check VAL_TUNING_SAMPLE_ROWS relative to the known validation positive rate."
            )
        del model
        gc.collect()
        return average_precision_score(y_val_sample, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def parse_seeds(spec: str) -> list[int]:
    """Parse a seed list like '42' or '42,7,13,99,23'."""
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        raise ValueError("--seeds produced no integers")
    seeds = [int(p) for p in parts]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicates: {seeds}")
    return seeds


def select_cost_optimal_threshold(y_val, probs_val, fn_cost: float, fp_cost: float) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_cost = 0.5, np.inf
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost = cost
            best_thresh = t
    return float(best_thresh)


def evaluate(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probs) if y_true.sum() > 0 else float("nan"),
        "auprc": average_precision_score(y_true, probs) if y_true.sum() > 0 else float("nan"),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": threshold,
    }


def run_train_mode(args) -> None:
    models_dir, results_dir = Path(args.out_dir[0]), Path(args.out_dir[1])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    exclude_cols = None if args.keep_infra_features else list(INFRA_FEATURE_COLS)

    print("[LOAD] Reading resampled training set (chunked, downcast to float32 as it goes)...")
    train_df, feature_cols = load_train_chunked(Path(args.train), args.target, exclude_cols)
    print(f"Using {len(feature_cols)} feature columns")

    X_train, y_train = train_df[feature_cols], train_df[args.target]

    print("[LOAD] Sampling validation for tuning (fixed prefix, kept in memory across trials)...")
    X_val_sample, y_val_sample = load_val_tuning_sample(
        Path(args.val), feature_cols, args.target, VAL_TUNING_SAMPLE_ROWS
    )

    print(f"[OPTUNA] Running {args.n_trials} trials optimizing AUPRC on the validation sample...")
    best_params = tune_with_optuna(X_train, y_train, X_val_sample, y_val_sample, n_trials=args.n_trials)
    print(f"[OPTUNA] Best params: {best_params}")
    del X_val_sample, y_val_sample
    gc.collect()

    # Multi-seed ensemble.  GBDTs are stochastic (feature/row subsampling), so
    # averaging a handful of independently seeded models is a cheap, reliable
    # variance reduction.  The FIRST seed is also written to lgbm_model.txt so
    # every downstream consumer (SHAP Stage 11, the ablation scripts) keeps
    # working unchanged; the full ensemble is saved separately for predict
    # mode to average.
    seeds = parse_seeds(args.seeds)
    print(f"[SEEDS] Training {len(seeds)} model(s) with seeds {seeds}")
    models = []
    for i, seed in enumerate(seeds, start=1):
        params = {**best_params, "objective": "binary", "metric": "average_precision",
                  "verbosity": -1, "random_state": seed}
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train, y_train)
        models.append(m)
        print(f"[SEEDS]   {i}/{len(seeds)} fitted (seed={seed})")
        if i == 1:
            m.booster_.save_model(str(models_dir / "lgbm_model.txt"))

    if len(models) > 1:
        for m, seed in zip(models, seeds):
            m.booster_.save_model(str(models_dir / f"lgbm_model_seed{seed}.txt"))
        print(f"[SEEDS] Saved {len(models)} boosters for ensemble averaging")

    del train_df, X_train, y_train
    gc.collect()

    with open(models_dir / "lgbm_feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "target": args.target}, f, indent=2)
    with open(models_dir / "lgbm_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open(models_dir / "lgbm_ensemble_seeds.json", "w") as f:
        json.dump({"seeds": seeds, "n_models": len(models),
                   "excluded_features": exclude_cols or []}, f, indent=2)

    print(f"\n[DONE -- TRAIN MODE] Model trained and saved.")
    print(f"[SAVE] {models_dir/'lgbm_model.txt'}")
    print(f"[SAVE] {models_dir/'lgbm_feature_cols.json'}")
    print(f"[SAVE] {models_dir/'lgbm_ensemble_seeds.json'}")
    print(f"       Run --mode predict as a separate, fresh command.")


def run_predict_mode(args) -> None:
    models_dir, results_dir = Path(args.out_dir[0]), Path(args.out_dir[1])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_cols_path = models_dir / "lgbm_feature_cols.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError(
            f"{feature_cols_path} not found -- run --mode train first, as a separate command."
        )
    with open(feature_cols_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    target = meta.get("target", args.target)

    with open(args.cost_matrix) as f:
        cost_matrix = json.load(f)

    print("[LOAD] Loading trained model(s) from disk (fresh process, no training history)...")
    ens_path = models_dir / "lgbm_ensemble_seeds.json"
    boosters = []
    if ens_path.exists():
        with open(ens_path) as f:
            ens_meta = json.load(f)
        for sd in ens_meta.get("seeds", []):
            p = models_dir / f"lgbm_model_seed{sd}.txt"
            if p.exists():
                boosters.append(lgb.Booster(model_file=str(p)))
        if boosters:
            print(f"[ENSEMBLE] Averaging {len(boosters)} boosters: {ens_meta.get('seeds')}")
    if not boosters:
        boosters = [lgb.Booster(model_file=str(models_dir / "lgbm_model.txt"))]
        print("[ENSEMBLE] Single model (no seed files found) -- not averaging")

    print("[STREAM] Predicting on FULL validation set (chunked) for threshold selection...")
    y_val_full, val_probs_full, val_serials, val_dates = stream_predict_ensemble(
        boosters, Path(args.val), feature_cols, target
    )
    threshold = select_cost_optimal_threshold(
        y_val_full, val_probs_full, cost_matrix["false_negative_cost"], cost_matrix["false_positive_cost"]
    )
    print(f"[THRESHOLD] Cost-optimal threshold on FULL validation: {threshold:.3f}")
    # Saved WITH keys (serial_number, date) and target -- Stage 8 needs val
    # probs (not just test) to fit/select its meta-learner, and needs a
    # reliable join key rather than assuming row order matches the LSTM's
    # full-fleet output, which is chunked/sorted differently.
    pd.DataFrame({
        "serial_number": decode_keys(val_serials), "date": decode_keys(val_dates),
        "lgbm_prob": val_probs_full, target: y_val_full,
    }).to_csv(results_dir / "lgbm_val_probs.csv", index=False)
    del y_val_full, val_probs_full, val_serials, val_dates
    gc.collect()

    print("[STREAM] Predicting on FULL test set (chunked) for final evaluation...")
    y_test_full, test_probs_full, test_serials, test_dates = stream_predict_ensemble(
        boosters, Path(args.test), feature_cols, target,
        ckpt_dir=(results_dir / "_ens_ckpt_test") if args.ensemble_ckpt else None,
    )
    if args.ensemble_ckpt:
        ck = results_dir / "_ens_ckpt_test"
        for f in ("acc.npy", "meta.npz"):
            try:
                (ck / f).unlink()
            except OSError:
                pass
        try:
            ck.rmdir()
        except OSError:
            pass
    metrics = evaluate(y_test_full, test_probs_full, threshold)
    print("[TEST METRICS]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(models_dir / "lgbm_threshold.json", "w") as f:
        json.dump({"threshold": threshold, "target": target, "feature_cols": feature_cols}, f, indent=2)
    pd.DataFrame({
        "serial_number": decode_keys(test_serials), "date": decode_keys(test_dates),
        "lgbm_prob": test_probs_full, target: y_test_full,
    }).to_csv(results_dir / "lgbm_test_probs.csv", index=False)
    pd.DataFrame([metrics]).to_csv(results_dir / "lgbm_metrics.csv", index=False)

    print(f"\n[DONE -- PREDICT MODE]")
    print(f"[SAVE] {models_dir/'lgbm_threshold.json'}")
    print(f"[SAVE] {results_dir/'lgbm_val_probs.csv'}")
    print(f"[SAVE] {results_dir/'lgbm_test_probs.csv'}")
    print(f"[SAVE] {results_dir/'lgbm_metrics.csv'}")


def main() -> None:
    print_environment_diagnostics()

    parser = argparse.ArgumentParser(description="Stage 6 -- two-mode, process-isolated LightGBM training/prediction")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"])
    parser.add_argument("--train", type=str, help="Required for --mode train")
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, help="Required for --mode predict")
    parser.add_argument("--cost_matrix", type=str, help="Required for --mode predict")
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds. More than one trains an "
                             "ensemble whose probabilities are averaged in predict mode. "
                             "The first seed is always written to lgbm_model.txt so "
                             "downstream consumers keep working.")
    parser.add_argument("--ensemble_ckpt", action="store_true",
                        help="Checkpoint the ensemble accumulator after each pass and resume "
                             "from it on a later run. Recommended for multi-seed ensembles "
                             "over very large test splits.")
    parser.add_argument("--keep_infra_features", action="store_true",
                        help=f"Keep the infrastructure identifiers {INFRA_FEATURE_COLS}. "
                             "Off by default: they are not SMART health signals, they rank "
                             "as the strongest SHAP driver, and removing them raises test "
                             "AUPRC by ~3.3%.")
    parser.add_argument("--out_dir", nargs=2, default=["models", "results"], metavar=("MODELS_DIR", "RESULTS_DIR"))
    args = parser.parse_args()

    if args.mode == "train":
        if not args.train:
            parser.error("--train is required for --mode train")
        run_train_mode(args)
    else:
        if not args.test or not args.cost_matrix:
            parser.error("--test and --cost_matrix are required for --mode predict")
        run_predict_mode(args)


if __name__ == "__main__":
    main()