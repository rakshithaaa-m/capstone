"""
Stage 16 -- Stacking study: can ANY combiner beat LightGBM alone?

Context
-------
Stage 8 ships `lgbm_only`. The meta-learner is a 2-coefficient logistic
regression on [lgbm_prob, rul_risk].  A single global coefficient cannot
express "trust RUL only when the drive looks near-failure" -- if RUL carries
signal in one regime and noise in another, a fixed blend can only average the
two and lose in both.

This script tests whether the LIMIT is the combiner or the RUL feature, by
trying progressively more expressive combiners on the SAME predictions:

  A. intercept-only          (should tie LightGBM exactly -- sanity check)
  B. logistic  [prob, risk]  (what Stage 8 does today)
  C. GBDT stacker            (can gate on regime)
  D. GBDT stacker + richer features (age, model, error_total, ...)

Correctness requirements
-------------------------
* Threshold selection on validation only.
* The combiner is fit on a FIT half and scored on a SELECT half, so its own
  score is not inflated by having seen the selection rows.
* Grouping is by physical drive: rows from one drive share a degradation
  trajectory and are strongly correlated.
* Report the gap to LightGBM with a drive-clustered bootstrap CI, not just a
  point estimate, because the differences we are chasing are ~1%.

Partner models (Amendment 03)
-----------------------------
The second score is pluggable.  ``--partner_csv`` is any per-row file with
``serial_number``, ``date`` and ``--partner_col``; ``--partner_kind`` says
whether that column is a remaining-life estimate in days (``rul``, mapped
through the horizon-bounded encoding exactly as before) or a failure
probability (``prob``, used as-is).  The defaults reproduce the original
LightGBM + LSTM study byte-for-byte (checked with ``--boot 400``).

Usage:
    python 16_stacking_study.py --out results/stacking_study.csv
    python 16_stacking_study.py --partner_csv results_fusion/pa_xgb_aft_val_preds.csv \\
        --partner_col aft_rul_pred --partner_kind rul \\
        --out results_fusion/stacking_xgb_aft.csv --boot 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260926


def load_split(results_dir: Path, partner_csv: Path,
               partner_col: str) -> pd.DataFrame:
    """Latest-known row per drive with both scores aligned."""
    import duckdb

    con = duckdb.connect(config={"memory_limit": "6GB",
                                 "temp_directory": str(Path.home() / "duck_spool")})
    try:
        df = con.execute(f"""
            WITH src AS (
                SELECT serial_number, date, lgbm_prob, hybrid_score, target
                FROM read_csv_auto('{(results_dir/'hybrid_val_scores.csv').resolve().as_posix()}')
            ),
            -- Latest-known assessment per drive.
            last AS (
                SELECT serial_number,
                       ARG_MAX(lgbm_prob, date)     AS prob,
                       ARG_MAX(hybrid_score, date)  AS hybrid_score
                FROM src GROUP BY 1
            ),
            -- Label the DRIVE, not the row.  `target` on a drive's last
            -- observed day is almost never 1 (a drive's final day sits
            -- outside the 14-day window), leaving ~70 positives and an AUPRC
            -- interval spanning a 5x range -- far too noisy to compare
            -- combiners.  "Ever fails in the horizon" is both the
            -- operationally meaningful question and has enough positives to
            -- measure.  Aggregated over ALL rows, independent of the collapse.
            ever AS (
                SELECT serial_number, MAX(target) AS ever_fail
                FROM src GROUP BY 1
            ),
            p AS (
                SELECT serial_number, ARG_MAX("{partner_col}", date) AS rul
                FROM read_csv_auto('{partner_csv.resolve().as_posix()}')
                GROUP BY 1
            )
            SELECT last.serial_number, last.prob, last.hybrid_score,
                   p.rul, ever.ever_fail,
                   p.serial_number IS NOT NULL AS has_partner
            FROM last
            JOIN ever ON ever.serial_number = last.serial_number
            LEFT JOIN p ON p.serial_number  = last.serial_number
        """).fetchdf()
    finally:
        con.close()
    # Every partner is compared on the SAME drives as LightGBM; a partner
    # that silently scores fewer drives would shift the baseline too.
    missing = int((~df["has_partner"]).sum())
    print(f"[LOAD] drives without a partner score: {missing:,} of {len(df):,}"
          + ("  (dropped)" if missing else ""))
    return df[df["has_partner"]].drop(columns="has_partner")


def features(prob: np.ndarray, rul: np.ndarray, horizon: float,
             kind: str = "rul") -> dict:
    """Derive combiner inputs from the two base scores.

    ``rul`` is the partner score: days of remaining life when ``kind`` is
    ``rul``, a failure probability when ``kind`` is ``prob``.  Non-finite
    values (a drive with no matched partner row) are clipped to 0 so the
    feature matrices stay finite.
    """
    rul = np.nan_to_num(np.asarray(rul, dtype=np.float64), nan=0.0,
                        posinf=0.0, neginf=0.0)
    prob = np.nan_to_num(np.asarray(prob, dtype=np.float64), nan=0.0,
                         posinf=1.0, neginf=0.0)
    if kind == "prob":
        risk = np.clip(rul, 0.0, 1.0)
        clipped = np.clip(risk, 1e-7, 1 - 1e-7)
        return {
            "prob": prob,
            "risk": risk,
            "prob_x_risk": prob * risk,
            "prob_squared": prob * prob,
            "risk_logit": np.log(clipped / (1 - clipped)),
            "prob_high": (prob > 0.5).astype(np.float64),
            "risk_high": (risk > 0.5).astype(np.float64),
        }
    risk = np.clip((horizon - rul) / horizon, 0.0, 1.0)
    hi = float(np.percentile(rul, 99)) if np.any(rul > 0) else 1.0
    return {
        "prob": prob,
        "risk": risk,
        # Interactions a linear model cannot form but a tree can exploit.
        "prob_x_risk": prob * risk,
        "prob_squared": prob * prob,
        "rul_log": np.log1p(np.clip(rul, 0, None)),
        # Non-linear reshapes of the two base signals.
        "prob_high": (prob > 0.5).astype(np.float64),
        "risk_high": (risk > 0.5).astype(np.float64),
        "rul_frac": rul / hi,
    }


def fit_predict(name: str, X_fit, y_fit, X_sel, params: dict | None = None):
    if name == "intercept_only":
        return np.full(len(X_sel), float(np.mean(y_fit)))
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(X_fit, y_fit)
        return m.predict_proba(X_sel)[:, 1]
    import lightgbm as lgb
    p = {"objective": "binary", "metric": "average_precision", "verbosity": -1,
         "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 200,
         "n_estimators": 400, "subsample": 0.8, "subsample_freq": 1,
         "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": SEED}
    if params:
        p.update(params)
    m = lgb.LGBMClassifier(**p)
    m.fit(X_fit, y_fit)
    return m.predict_proba(X_sel)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--horizon_days", type=float, default=14.0)
    ap.add_argument("--out", default="results/stacking_study.csv")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--partner_csv", default=None,
                    help="per-row partner scores (default: <results_dir>/"
                         "lstm_full_val_preds.csv)")
    ap.add_argument("--partner_col", default="lstm_rul_pred")
    ap.add_argument("--partner_kind", choices=["rul", "prob"], default="rul")
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score as AP
    from sklearn.model_selection import GroupShuffleSplit

    results = Path(args.results_dir)
    cfg = json.loads((Path("models") / "hybrid_config.json").read_text())
    horizon = float(cfg.get("horizon_days", args.horizon_days))

    print("[LOAD] one row per drive, ever-fail label ...\n")
    partner_csv = (Path(args.partner_csv) if args.partner_csv
                   else results / "lstm_full_val_preds.csv")
    print(f"[PARTNER] {partner_csv}  column={args.partner_col}  "
          f"kind={args.partner_kind}")
    df = load_split(results, partner_csv, args.partner_col)
    order = np.argsort(df["serial_number"].to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    y = df["ever_fail"].astype(int).to_numpy()
    groups = df["serial_number"].to_numpy()
    prob = df["prob"].to_numpy(dtype=np.float64)
    rul = df["rul"].to_numpy(dtype=np.float64)
    df = None
    print(f"[LOAD] {len(y):,} drives, {int(y.sum()):,} ever-fail "
          f"({100 * y.mean():.4f}%)")

    F = features(prob, rul, horizon, args.partner_kind)

    # Deterministic split. GroupShuffleSplit is seeded, but DuckDB does not
    # guarantee row order, so an unstable input order silently reshuffles
    # which drives land in each half -- with few positives that flips the
    # AUPRC ranking. load_split() sorts by serial so the split is reproducible.

    # Drive-level fit/select split so the combiner never scores rows it saw.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    fit_idx, sel_idx = next(gss.split(F["prob"], y, groups))
    print(f"[SPLIT] fit {len(fit_idx):,} / select {len(sel_idx):,} drives, "
          f"select positives {int(y[sel_idx].sum())}")

    specs = [
        ("A. intercept only (sanity)", ["prob"]),
        ("B. logistic [prob, risk]", ["prob", "risk"]),
        ("C. GBDT stacker [prob, risk]", ["prob", "risk"]),
        ("D. GBDT + engineered",
         ["prob", "risk", "prob_x_risk", "rul_log", "prob_high", "risk_high",
          "rul_frac"] if args.partner_kind == "rul" else
         ["prob", "risk", "prob_x_risk", "risk_logit", "prob_high",
          "risk_high"]),
        ("E. GBDT prob-only (control)", ["prob"]),
    ]
    rows = []
    store = {}
    base_sel = AP(y[sel_idx], F["prob"][sel_idx])
    print(f"\n[BASELINE] LightGBM alone on the selection half: AUPRC {base_sel:.5f}")
    print(f"[BASELINE] positives in selection half: {int(y[sel_idx].sum()):,}")
    print("[RESULT]  combiner                 AUPRC      vs LightGBM")

    for name, cols in specs:
        X = np.column_stack([F[c] for c in cols])
        if name.startswith("A"):
            preds = np.full(len(sel_idx), float(np.mean(y[fit_idx])))
        elif name.startswith("B"):
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(class_weight="balanced", max_iter=1000)
            m.fit(X[fit_idx], y[fit_idx])
            preds = m.predict_proba(X[sel_idx])[:, 1]
        else:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(objective="binary", metric="average_precision",
                                    verbosity=-1, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=200, n_estimators=400,
                                    subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, reg_lambda=1.0,
                                    random_state=SEED)
            m.fit(X[fit_idx], y[fit_idx])
            preds = m.predict_proba(X[sel_idx])[:, 1]
        ap_ = float(AP(y[sel_idx], preds))
        delta = ap_ - base_sel
        store[name] = preds
        rows.append({"combiner": name, "features": "+".join(cols),
                     "auprc": ap_, "lgbm_only_auprc": base_sel, "delta": delta,
                     "n_features": len(cols)})
        print(f"          {name:32s} {ap_:.5f}  {delta:+.5f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n[SAVE] {args.out}")

    # ---- Does RUL carry information the classifier does NOT already have? --
    # The combiner comparison above is confounded: a GBDT fit on two derived
    # columns is a far weaker learner than the 212-feature LightGBM, so it can
    # lose on its own terms (spec E, prob-only, loses without RUL anywhere in
    # the input).  Losing therefore does not by itself prove RUL is useless.
    #
    # The confound-free question is a CONDITIONAL one: within a narrow band of
    # LightGBM probability -- where the classifier is already calibrated and
    # has extracted what it can -- does the RUL-derived risk score still rank
    # failures above survivors?  AUROC within each band needs no model fitting
    # at all, so it cannot overfit and cannot inherit the combiner's weakness.
    from sklearn.metrics import roc_auc_score as AUC

    risk_all = F["risk"]
    sel_pos = y[sel_idx] == 1
    print("\n[CONDITIONAL] does the partner add signal ON TOP of LightGBM probability?")
    marginal = AUC(y[sel_idx], risk_all[sel_idx])
    print(f"[CONDITIONAL] marginal AUROC of partner risk alone: {marginal:.4f}")
    bands = []
    qs = np.quantile(prob[sel_idx], np.linspace(0, 1, 11))
    for b in range(10):
        lo, hi = qs[b], qs[b + 1]
        m = (prob[sel_idx] >= lo) & (prob[sel_idx] <= hi if b == 9 else prob[sel_idx] < hi)
        npos = int(sel_pos[m].sum())
        if npos < 5 or npos == int(m.sum()):
            bands.append({"band": f"P{b*10}-{(b+1)*10}", "n": int(m.sum()),
                          "positives": npos, "aulroc_risk": None})
            continue
        a = AUC(y[sel_idx][m], risk_all[sel_idx][m])
        bands.append({"band": f"P{b*10}-{(b+1)*10}", "n": int(m.sum()),
                      "positives": npos, "aulroc_risk": a})
        print(f"[CONDITIONAL]   prob {b*10:>2}-{(b+1)*10:>2}%  n={int(m.sum()):>7,}  "
              f"pos={npos:>4}  AUROC(partner)={a:.4f}")
    bd = pd.DataFrame(bands)
    valid = bd["aulroc_risk"].dropna()
    if len(valid):
        print(f"[CONDITIONAL] bands where the partner ranks failures correctly: "
              f"{int((valid > 0.5).sum())}/{len(valid)}   "
              f"median AUROC {valid.median():.4f}")
        # The median weights every band equally, but positives are not spread
        # evenly: on the LSTM run the top decile held 1,178 of 1,353.  The
        # positive-weighted mean is the Amendment 03 screening statistic.
        w = bd.loc[valid.index, "positives"].to_numpy(dtype=np.float64)
        print(f"[CONDITIONAL] positive-weighted mean AUROC "
              f"{float(np.average(valid.to_numpy(), weights=w)):.4f}")
    bd.to_csv(args.out.replace(".csv", "_conditional.csv"), index=False)

    # ---- Paired drive-clustered bootstrap on the winner -------------------
    # A point estimate is not evidence.  Resample DRIVES (rows within a drive
    # share a trajectory) and recompute both models on the same resample, so
    # the comparison is paired and the interval reflects real drive-to-drive
    # variance rather than row-level noise.
    best_row = out.loc[out["auprc"].idxmax()]
    # Bootstrap the WINNER'S ACTUAL PREDICTIONS.  Reading a feature column off
    # the matrix instead would compare LightGBM against itself (column 0 of
    # every spec is `prob`), which silently yields a gap of exactly zero.
    ps = store[best_row["combiner"]]
    ys = y[sel_idx]
    gb = groups[sel_idx]
    pb = prob[sel_idx]
    uq = np.unique(gb)
    bounds = np.zeros(len(uq) + 1, dtype=np.int64)
    cnt = np.bincount(np.searchsorted(uq, gb), minlength=len(uq))
    np.cumsum(cnt, out=bounds[1:])
    rng = np.random.default_rng(SEED)
    gaps = []
    for _ in range(args.boot):
        pick = rng.integers(0, len(uq), len(uq))
        idx = np.concatenate([np.arange(bounds[q], bounds[q + 1]) for q in pick])
        yy = ys[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        gaps.append(AP(yy, ps[idx]) - AP(yy, pb[idx]))
    gaps = np.asarray(gaps)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    wins = 100.0 * (gaps > 0).mean()
    print(f"\n[BOOTSTRAP] best combiner vs LightGBM, drive-clustered, B={len(gaps)}")
    print(f"[BOOTSTRAP] gap AUPRC {gaps.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"[BOOTSTRAP] stacker wins in {wins:.1f}% of resamples")
    out.attrs["gap_ci"] = (float(lo), float(hi))
    out.to_csv(args.out, index=False)
    pd.DataFrame([{"best_combiner": best_row["combiner"],
                   "gap_mean": float(gaps.mean()), "ci_lo": float(lo),
                   "ci_hi": float(hi), "wins_pct": float(wins),
                   "n_positives_select": int(y[sel_idx].sum()),
                   "base_auprc": base_sel}]).to_csv(
        args.out.replace(".csv", "_bootstrap.csv"), index=False)

    if lo <= 0 <= hi:
        print("\n[VERDICT] Gap is NOT distinguishable from noise (CI spans 0).")
        print("[VERDICT] Do not ship a fusion. Report that the limit is the partner")
        print("[VERDICT] score, not the combiner -- a linear blend and two GBDT")
        print("[VERDICT] stackers were all tested and none beat LightGBM reliably.")
    elif lo > 0:
        print(f"\n[VERDICT] Gap EXCLUDES zero: {best_row['combiner']} genuinely beats")
        print(f"[VERDICT] LightGBM by {gaps.mean():.5f} AUPRC. Ship it.")
    else:
        print(f"\n[VERDICT] Gap excludes zero in the WRONG direction: the best")
        print("[VERDICT] combiner is reliably worse. Report that.")


if __name__ == "__main__":
    main()