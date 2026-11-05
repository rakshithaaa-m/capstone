"""Survival (urgency) model refitted on Model 2's training window.

Same estimator, features, label form and Optuna search as 20_train_xgb_aft.py, on
Model 2's data: the undersampled training rows of models_retrain2026/A (to
2025-06-16) and a tuning sample of its validation period (2025-07-01..12-17).

Labels are cut at each split's end: a failure counts as an event only if it
happens by the split's last day; otherwise the drive is censored at its last
observation inside the split.  No label uses data from after the split.

    python 37_urgency_model2.py build      # training rows, tuning sample, validation latest rows
    python 37_urgency_model2.py train      # Optuna (30 trials) + final fit -> models/urgency/
    python 37_urgency_model2.py predict    # validation reference fleet -> models/urgency/pa_xgb_aft_val_preds.csv
    python 37_urgency_model2.py all
Then `31_deploy_2026.py urgency dashboard` and `35_model2_eval.py urgency`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

TRAIN_END = "2025-06-16"
VAL_START, VAL_END = "2025-07-01", "2025-12-17"
SOURCES = [Path("data/processed") / f for f in ("features_train.parquet", "features_val.parquet",
                                                 "features_test.parquet")]
TRAIN_ROWS = Path("data/processed/retrain2026/train_resampled_A.parquet")
WORK = Path("results_model2/_work/urgency")
MODEL_DIR = Path("models/urgency")
TUNE_BUCKETS = 11          # hash(serial) % 11 == 0: about 5M validation rows, whole drives
FEATS = json.load(open("models/lgbm_feature_cols.json"))["feature_cols"]


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


def _s20():
    spec = importlib.util.spec_from_file_location("s20", "20_train_xgb_aft.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s20"] = mod
    spec.loader.exec_module(mod)
    return mod


def _src(lo: str | None, hi: str) -> str:
    files = ", ".join(f"'{p.as_posix()}'" for p in SOURCES)
    cond = f"CAST(date AS DATE) <= DATE '{hi}'" + (f" AND CAST(date AS DATE) >= DATE '{lo}'" if lo else "")
    return f"read_parquet([{files}]) WHERE {cond}"


def build() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    con = _duck()
    cols = ", ".join(f't."{c}"' for c in FEATS)
    jobs = {
        # Model 2's exact training rows, with remaining-life fields and last day seen in the window
        "train.parquet": f"""
            WITH src AS (SELECT serial_number, CAST(date AS DATE) d, rul_days, is_censored FROM {_src(None, TRAIN_END)}),
                 last AS (SELECT serial_number, max(d) last_d FROM src GROUP BY 1)
            SELECT {cols}, t.serial_number, CAST(t.date AS DATE) d, s.rul_days, s.is_censored, l.last_d
            FROM read_parquet('{TRAIN_ROWS.as_posix()}') t
            JOIN src s ON s.serial_number = t.serial_number AND s.d = CAST(t.date AS DATE)
            JOIN last l ON l.serial_number = t.serial_number""",
        # tuning sample: whole drives, about 5M validation rows
        "tune.parquet": f"""
            WITH v AS (SELECT * FROM {_src(VAL_START, VAL_END)}),
                 last AS (SELECT serial_number, max(CAST(date AS DATE)) last_d FROM v GROUP BY 1)
            SELECT {cols}, t.serial_number, CAST(t.date AS DATE) d, t.rul_days, t.is_censored, l.last_d
            FROM v t JOIN last l USING (serial_number)
            WHERE hash(t.serial_number) % {TUNE_BUCKETS} = 0""",
        # reference fleet: each drive's latest validation row
        "val_latest.parquet": f"""
            SELECT {cols}, t.serial_number, CAST(t.date AS DATE) d
            FROM (SELECT * FROM {_src(VAL_START, VAL_END)}) t
            QUALIFY row_number() OVER (PARTITION BY t.serial_number ORDER BY t.date DESC) = 1""",
    }
    for name, sql in jobs.items():
        dst = WORK / name
        if not dst.exists():
            con.execute(f"COPY ({sql}) TO '{dst.with_suffix('.tmp').as_posix()}' (FORMAT parquet)")
            dst.with_suffix(".tmp").replace(dst)
        print(f"[BUILD] {name}: {pq.ParquetFile(dst).metadata.num_rows:,} rows", flush=True)


def _rows(path: Path, end: str):
    """Features and AFT bounds (days + 1), with every label cut at `end`."""
    t = pq.read_table(path)
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                         for c in FEATS])
    d = t.column("d").to_numpy().astype("datetime64[D]")
    last = t.column("last_d").to_numpy().astype("datetime64[D]")
    rul = t.column("rul_days").to_numpy(zero_copy_only=False).astype(np.float64)
    failing = ~t.column("is_censored").to_numpy(zero_copy_only=False).astype(bool)
    del t
    fail_day = d + np.where(failing, np.nan_to_num(rul), 0).astype(np.int64).astype("timedelta64[D]")
    event = failing & (fail_day <= np.datetime64(end))
    keep = ~(failing & (rul < 0))            # rows after a recorded failure (returned / reused drives)
    observed = (last - d).astype(np.float64)
    lower = np.where(event, rul, observed) + 1.0
    upper = np.where(event, rul + 1.0, np.inf)
    info = {"rows": int(len(d)), "dropped_post_failure": int((~keep).sum()),
            "events": int(event[keep].sum()), "censored_share": float(1 - event[keep].mean()),
            "failing_after_end_censored": int((failing & ~event & keep).sum())}
    return X[keep], lower[keep].astype(np.float32), upper[keep].astype(np.float32), info


def train(n_trials: int) -> None:
    import xgboost as xgb
    s20 = _s20()
    X, lo, hi, i_tr = _rows(WORK / "train.parquet", TRAIN_END)
    print(f"[LOAD] train {i_tr}", flush=True)
    d_tr = s20.dmatrix(X, lo, hi)
    del X
    X, lo, hi, i_va = _rows(WORK / "tune.parquet", VAL_END)
    print(f"[LOAD] tune {i_va}", flush=True)
    d_va = s20.dmatrix(X, lo, hi)
    del X
    best, rounds = s20.tune(d_tr, d_va, n_trials)
    print(f"[OPTUNA] best {best}  rounds {rounds}", flush=True)
    params = {**s20.BASE_PARAMS, **best, "seed": s20.SEED}
    bst = xgb.train(params, d_tr, num_boost_round=rounds, evals=[(d_va, "val")], verbose_eval=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bst.save_model(MODEL_DIR / "pa_xgb_aft.json")
    (MODEL_DIR / "pa_xgb_aft_params.json").write_text(json.dumps(
        {"params": params, "num_boost_round": rounds, "n_trials": n_trials,
         "train_window": ["2013-04-10", TRAIN_END], "tune_window": [VAL_START, VAL_END],
         "train": i_tr, "tune": i_va, "labels": "events only up to each split's end; otherwise censored",
         "time_unit": "rul_days + 1 (prediction file is back in days)", "script": "37_urgency_model2.py"},
        indent=2))
    print(f"[SAVE] {MODEL_DIR / 'pa_xgb_aft.json'}", flush=True)


def predict() -> None:
    import pandas as pd
    import xgboost as xgb
    bst = xgb.Booster(model_file=str(MODEL_DIR / "pa_xgb_aft.json"))
    t = pq.read_table(WORK / "val_latest.parquet")
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                         for c in FEATS])
    pred = np.clip(bst.inplace_predict(X) - 1.0, 0.0, None)
    out = MODEL_DIR / "pa_xgb_aft_val_preds.csv"
    pd.DataFrame({"serial_number": t.column("serial_number").to_pandas(),
                  "date": pd.to_datetime(t.column("d").to_pandas()).dt.strftime("%Y-%m-%d"),
                  "aft_rul_pred": pred.astype(np.float32)}).to_csv(out, index=False, lineterminator="\n")
    print(f"[SAVE] {out}  {len(pred):,} drives (latest validation row each)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "train", "predict", "all"])
    ap.add_argument("--n_trials", type=int, default=30)
    args = ap.parse_args()
    for st in (["build", "train", "predict"] if args.stage == "all" else [args.stage]):
        {"build": build, "train": lambda: train(args.n_trials), "predict": predict}[st]()


if __name__ == "__main__":
    main()
