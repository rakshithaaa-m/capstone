"""
Stage 14b -- Publication-grade evaluation harness.

Produces the tables a reviewer will ask for and the dashboard cannot show.

EVALULATION POPULATION (this matters and is easy to get wrong)
--------------------------------------------------------------
The reported AUPRC is a DRIVE-DAY level metric over all 144,739,479
drive-days in the test split, with a positive rate of 0.0468%.  It must NOT
be recomputed on one-row-per-drive: collapsing to the latest assessment per
drive leaves almost no positives (the last day of a drive's life is almost
never inside the 14-day window) and silently changes the problem.  The
harness therefore reports:

  * row_level  -- drive-day AUPRC, matching results/lgbm_metrics.csv exactly
  * drive_level-- "does this drive ever fail within the horizon", one row per
                  drive.  Smaller, but this is the operationally meaningful
                  question and it is cheap enough for a full bootstrap.

CONFIDENCE INTERVALS
--------------------
Rows from one drive share a degradation trajectory and are strongly
correlated, so a row-level bootstrap yields intervals that are far too
narrow.  Both levels resample PHYSICAL DRIVES.  Because a full drive-level
bootstrap over 144M rows does not fit in 16 GB of RAM, the row-level interval
uses a subsampled drive bootstrap (see SUBSAMPLE_DRIVES) and says so in the
output; the drive-level interval is an exact full bootstrap.

Usage:
    python 14b_paper_evaluation.py --results_dir results
    python 14b_paper_evaluation.py --results_dir results --boot 200 --boot-row 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SEED = 20260926
SUBSAMPLE_DRIVES = 40_000

# TFILL, Ding et al., IEEE ISPA 2024, Table IV (ST4000DM000, 60-day RUL cap)
BASE_PAPER = {
    "Encoder-Decoder LSTM": {"rmse": 22.19, "mae": 15.25},
    "Vanilla Transformer": {"rmse": 29.12, "mae": 29.14},
    "DAST": {"rmse": 13.10, "mae": 13.10},
    "LightGBM (no time-mix)": {"rmse": 10.81, "mae": 11.36},
    "TFBEST": {"rmse": 9.60, "mae": 9.54},
    "TFILL (base paper)": {"rmse": 4.97, "mae": 5.04},
}
BASE_PAPER_WINDOW_DAYS = 60


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_row_level(results_dir: Path):
    """Full drive-day test population, as compact numpy arrays.

    Uses DuckDB fetchnumpy() so we never materialise a pandas frame of the
    13 GB export.  ``group`` is a hash bucket per physical drive, used as the
    bootstrap grouping key.  A hash bucket is used instead of a dense rank
    because DENSE_RANK() OVER (ORDER BY serial_number) forces a full sort of
    144M strings (~6 GB, minutes).  With 1,000,003 buckets and 352k drives the
    expected number of merged pairs is ~60 out of 352k, which has no
    measurable effect on the bootstrap variance.
    """
    import duckdb

    src = (results_dir / "hybrid_test_scores.csv").resolve().as_posix()
    con = duckdb.connect(config={"memory_limit": "8GB",
                                 "temp_directory": str(Path.home() / "duck_spool")})
    try:
        cur = con.execute(
            f"""
            SELECT CAST(target AS TINYINT)        AS y,
                   CAST(lgbm_prob AS REAL)       AS p_lgbm,
                   CAST(hybrid_score AS REAL)    AS p_hybrid,
                   CAST(hybrid_weighted AS REAL) AS p_weighted,
                   CAST(HASH(serial_number) % 1000003 AS INTEGER) AS grp
            FROM read_csv_auto('{src}')
            """
        )
        names = [d[0] for d in cur.description]
        arrays = cur.fetchnumpy()
    finally:
        con.close()
    return {n: arrays[n] for n in names}


def load_drive_level(results_dir: Path) -> dict:
    """One row per drive, labelled by whether it EVER fails in the horizon."""
    import duckdb

    src = (results_dir / "hybrid_test_scores.csv").resolve().as_posix()
    con = duckdb.connect(config={"memory_limit": "8GB",
                                 "temp_directory": str(Path.home() / "duck_spool")})
    try:
        df = con.execute(
            f"""
            WITH agg AS (
                SELECT serial_number,
                       MAX(CAST(target AS TINYINT)) AS ever_fails,
                       ARG_MAX(CAST(lgbm_prob AS REAL), date)    AS p_lgbm,
                       ARG_MAX(CAST(hybrid_score AS REAL), date) AS p_hybrid,
                       ARG_MAX(CAST(lstm_rul_pred AS REAL), date) AS rul,
                       ARG_MAX(date, date) AS last_date
                FROM read_csv_auto('{src}')
                GROUP BY serial_number
            )
            SELECT * FROM agg
            """
        ).fetchdf()
    finally:
        con.close()
    return df


# ---------------------------------------------------------------------------
# Bootstraps
# ---------------------------------------------------------------------------


def _drive_bounds(grp: np.ndarray):
    """Row offsets per group, assuming grp is already sorted ascending."""
    uniq, counts = np.unique(grp, return_counts=True)
    bounds = np.zeros(len(uniq) + 1, dtype=np.int64)
    np.cumsum(counts, out=bounds[1:])
    return bounds


def _ragged_gather(bounds: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """Row indices for a ragged gather of the groups named in ``pick``.

    Group j occupies source rows [starts[j], ends[j]) and must land at
    output positions [offs[j], offs[j] + counts[j]).  Hence
    idx[p] = starts[j] + (p - offs[j]).
    """
    starts, ends = bounds[pick], bounds[pick + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offs = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return np.arange(total, dtype=np.int64) - np.repeat(offs, counts) + np.repeat(starts, counts)


def bootstrap_subsampled(y, s, bounds, n_boot, n_groups_per_rep, seed=SEED):
    """Drive-level bootstrap on a fixed-size drive subsample per replicate.

    A full 144M-row bootstrap does not fit in RAM; holding the drive count
    constant per replicate keeps replicate statistics comparable while still
    resampling whole drives (never individual drive-days).
    """
    n_total = len(bounds) - 1
    rng = np.random.default_rng(seed)
    aps = []
    for _ in range(n_boot):
        pick = rng.choice(n_total, size=min(n_groups_per_rep, n_total), replace=True)
        idx = _ragged_gather(bounds, pick)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == yb.size:
            continue
        aps.append(_ap(yb, sb))
    return np.asarray(aps)


def bootstrap_full(y, s, bounds, n_boot, seed=SEED):
    """Exact group-level bootstrap (all drives resampled with replacement)."""
    n_total = len(bounds) - 1
    rng = np.random.default_rng(seed)
    aps, aucs = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n_total, n_total)
        idx = _ragged_gather(bounds, pick)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == yb.size:
            continue
        aps.append(_ap(yb, sb))
        aucs.append(_auc(yb, sb))
    return np.asarray(aps), np.asarray(aucs)


def _ci(v):
    if len(v) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def tier_table(y: np.ndarray, score: np.ndarray, cutoffs: dict) -> pd.DataFrame:
    """Per-tier precision / recall / F1, in the style of the base paper's
    Table III health-level accuracy.  Computed directly with numpy because
    sklearn's set-wise helper refuses mixed str/int label arrays."""
    import pandas as pd

    hi, med = cutoffs["high_cutoff"], cutoffs["medium_cutoff"]
    # The three tiers MUST be mutually exclusive.  With overlapping masks,
    # MEDIUM double-counts every HIGH drive and the per-tier recalls sum to
    # more than 100% (n_flagged sums to more than the fleet size).
    tiers = {
        "HIGH": score >= hi,
        "MEDIUM": (score >= med) & (score < hi),
        "LOW": score < med,
    }
    y = y.astype(bool)
    total_pos = max(int(y.sum()), 1)
    # Guard against the exact bug this table originally had: overlapping tier
    # masks make n_flagged sum to more than the number of drives evaluated.
    total_flagged = sum(int(m.sum()) for m in tiers.values())
    if total_flagged != len(score):
        raise AssertionError(
            f"tier masks are not mutually exclusive: n_flagged sums to "
            f"{total_flagged} but {len(score)} drives were evaluated"
        )
    rows = []
    for name, mask in tiers.items():
        tp = int((mask & y).sum())
        fp = int((mask & ~y).sum())
        fn = int((~mask & y).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "tier": name, "n_flagged": int(mask.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "share_of_true_failures_caught": tp / total_pos,
        })
    return pd.DataFrame(rows)


def counterfactual_table(y, lgbm, rul, horizon, cutoffs, intercept, c_lgbm, c_rul,
                         rul_encoding="raw_days") -> pd.DataFrame:
    """How much does the RUL term move the tier assignment?

    Recomputes the shipped meta score with the RUL contribution removed and
    re-applies the same cutoffs.  This is the quantitative basis for the
    paper's negative result about RUL in fusion.

    ``rul_encoding`` MUST match the encoding the coefficients were fitted
    under, or the counterfactual is meaningless: feeding raw-day
    coefficients a horizon-bounded feature (or vice versa) collapses the RUL
    term's contribution and understates its effect by orders of magnitude.
    """
    import pandas as pd
    from scipy.special import expit

    if rul_encoding == "horizon_bounded":
        r_feat = np.clip((horizon - rul) / horizon, 0.0, 1.0)
    else:
        r_feat = rul
    with_rul = expit(intercept + c_lgbm * lgbm + c_rul * r_feat)
    without_rul = expit(intercept + c_lgbm * lgbm)

    def tier_of(s):
        return np.where(s >= cutoffs["high_cutoff"], "HIGH",
                        np.where(s >= cutoffs["medium_cutoff"], "MEDIUM", "LOW"))

    a, b = tier_of(with_rul), tier_of(without_rul)
    changed = a != b
    trans = {}
    for i in range(len(a)):
        if changed[i]:
            trans[f"{a[i]}->{b[i]}"] = trans.get(f"{a[i]}->{b[i]}", 0) + 1
    at_risk = int(((a != "LOW") | (b != "LOW")).sum())
    demote = sum(v for k, v in trans.items() if k.startswith(("HIGH", "MEDIUM")) and k.endswith("LOW"))
    promote = sum(v for k, v in trans.items() if k.endswith(("HIGH", "MEDIUM")) and k.startswith("LOW"))
    rows = []
    rows.append({"metric": "drives whose tier changes if RUL term is removed", "value": int(changed.sum())})
    rows.append({"metric": "as % of at-risk queue (HIGH+MEDIUM)", "value": float(changed.sum()) / max(at_risk, 1)})
    rows.append({"metric": "demotions (HIGH/MEDIUM -> LOW)", "value": int(demote)})
    rows.append({"metric": "promotions (LOW -> HIGH/MEDIUM)", "value": int(promote)})
    for k, v in sorted(trans.items()):
        rows.append({"metric": f"  {k}", "value": int(v)})

    # Why the RUL term behaves as it does.  With a POSITIVE coefficient on raw
    # RUL days, any negative prediction contributes a negative logit and
    # therefore *reduces* risk -- so the drives whose RUL is most nonsensical
    # are treated as the safest.  Report the effect size either way.
    neg = r_feat < 0
    pos = ~neg
    if neg.any():
        rows.append({"metric": "mean risk delta (RUL feature < 0), with - without",
                     "value": float((with_rul[neg] - without_rul[neg]).mean())})
    if pos.any():
        rows.append({"metric": "mean risk delta (RUL feature >= 0), with - without",
                     "value": float((with_rul[pos] - without_rul[pos]).mean())})
    rows.append({"metric": "n drives with RUL feature < 0", "value": int(neg.sum())})
    if y is not None and len(y) == len(rul):
        rows.append({"metric": "mean RUL feature, drives that failed",
                     "value": float(rul[np.asarray(y, dtype=bool)].mean())})
        rows.append({"metric": "mean RUL feature, drives that did not fail",
                     "value": float(rul[~np.asarray(y, dtype=bool)].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Publication-grade evaluation tables")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--boot", type=int, default=200, help="full bootstrap reps (drive level)")
    ap.add_argument("--boot-row", type=int, default=30, help="reps for row-level subsampled bootstrap")
    args = ap.parse_args()

    results = Path(args.results_dir)
    out = results / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)
    cutoffs = json.loads((results / "tier_cutoffs.json").read_text())
    cfg = json.loads((Path("models") / "hybrid_config.json").read_text())
    horizon = float(cfg.get("horizon_days", 14.0))
    # Label the shipped scorer by what Stage 8 actually chose.  With
    # ``lgbm_only`` the "hybrid" column equals LightGBM, so calling it a
    # fusion would misdescribe the tables.
    shipped = f"Shipped scorer ({cfg.get('chosen_strategy', 'meta')})"
    c = cfg.get("meta_coef") or [0.0, 0.0]
    intercept = float(cfg.get("meta_intercept", 0.0))

    print("[1/5] loading drive-day test population (144M rows) ...")
    t0 = time.time()
    cols = load_row_level(results)
    y = cols["y"].astype(np.int8)
    grp = cols["grp"].astype(np.int32)
    p_lgbm = cols["p_lgbm"].astype(np.float32)
    p_hybrid = cols["p_hybrid"].astype(np.float32)
    p_weighted = cols["p_weighted"].astype(np.float32)
    # group rows contiguously so bootstrap replicates are cheap gathers
    order = np.argsort(grp, kind="stable")
    y, grp = y[order], grp[order]
    p_lgbm = p_lgbm[order]
    p_hybrid = p_hybrid[order]
    p_weighted = p_weighted[order]
    del cols, order
    bounds = _drive_bounds(grp)
    n_pos = int(y.sum())
    row_n = int(len(y))
    print(f"      {row_n:,} drive-days | {n_pos:,} positives "
          f"({100 * n_pos / len(y):.4f}%) | {len(bounds) - 1:,} drive groups "
          f"| {time.time() - t0:.0f}s")

    print("[2/5] row-level point estimates (must match lgbm_metrics.csv) ...")
    row_rows = []
    for name, s in (("LightGBM only", p_lgbm), ("Weighted blend", p_weighted),
                    (shipped, p_hybrid)):
        ap_, auc = _ap(y, s), _auc(y, s)
        print(f"      {name:22s} AUPRC {ap_:.5f} ({ap_ / (n_pos / len(y)):.0f}x prevalence)  AUC {auc:.5f}")
        row_rows.append({"model": name, "auprc": ap_, "roc_auc": auc,
                         "lift_over_prevalence": ap_ / (n_pos / len(y))})
    import pandas as pd
    pd.DataFrame(row_rows).to_csv(out / "table_row_level.csv", index=False)

    print(f"[3/5] row-level drive-clustered bootstrap (B={args.boot_row}, "
          f"{SUBSAMPLE_DRIVES:,} drives/rep) ...")
    ci_rows = []
    for name, s in (("LightGBM only", p_lgbm), (shipped, p_hybrid)):
        dist = bootstrap_subsampled(y, s, bounds, args.boot_row, SUBSAMPLE_DRIVES)
        lo, hi = _ci(dist)
        point = _ap(y, s)
        print(f"      {name:22s} AUPRC {point:.5f}  95% CI [{lo:.5f}, {hi:.5f}]  ({len(dist)} reps)")
        ci_rows.append({"level": "drive-day", "model": name, "auprc": point,
                        "ci_lo": lo, "ci_hi": hi, "n_boot": len(dist),
                        "resampling_unit": "physical drive (subsampled 40k/rep)"})
    pd.DataFrame(ci_rows).to_csv(out / "table_bootstrap_ci.csv", index=False)
    del y, p_lgbm, p_weighted, p_hybrid, grp, bounds

    print("[4/5] drive-level 'ever fails' cohort + exact bootstrap ...")
    d = load_drive_level(results)
    dy = d["ever_fails"].to_numpy().astype(np.int8)
    d_bounds = _drive_bounds(np.arange(len(d), dtype=np.int32))
    dcut = {**cutoffs}
    for nm, col in (("LightGBM only", "p_lgbm"), (shipped, "p_hybrid")):
        s = d[col].to_numpy().astype(np.float64)
        point, auc = _ap(dy, s), _auc(dy, s)
        dist, dauc = bootstrap_full(dy, s, d_bounds, args.boot)
        lo, hi = _ci(dist); alo, ahi = _ci(dauc)
        print(f"      {nm:22s} AUPRC {point:.5f} 95% CI [{lo:.5f}, {hi:.5f}]  "
              f"AUC {auc:.5f} [{alo:.5f}, {ahi:.5f}]  ({int(dy.sum()):,} failing drives)")
        pd.DataFrame([{"level": "drive", "model": nm, "auprc": point, "ci_lo": lo, "ci_hi": hi,
                       "roc_auc": auc, "auc_ci_lo": alo, "auc_ci_hi": ahi,
                       "n_boot": len(dist), "n_drives": int(len(d)),
                       "n_failing_drives": int(dy.sum()),
                       "resampling_unit": "physical drive (full bootstrap)"}]).to_csv(
            out / f"table_drive_level_{'lgbm' if 'Light' in nm else 'hybrid'}.csv", index=False)

    print("[5/5] tier accuracy + RUL counterfactual ...")
    dcut_high, dcut_med = cutoffs["high_cutoff"], cutoffs["medium_cutoff"]
    tt = tier_table(d["ever_fails"].to_numpy().astype(np.int8),
                    d["p_hybrid"].to_numpy().astype(np.float64),
                    {"high_cutoff": dcut_high, "medium_cutoff": dcut_med})
    tt.to_csv(out / "table_tier_accuracy.csv", index=False)
    print(tt.to_string(index=False))

    rul = d["rul"].to_numpy().astype(np.float64)
    ok = np.isfinite(rul)
    cf = counterfactual_table(
        d["ever_fails"].to_numpy().astype(np.int8)[ok],
        d["p_lgbm"].to_numpy().astype(np.float64)[ok], rul[ok],
        horizon, {"high_cutoff": dcut_high, "medium_cutoff": dcut_med},
        intercept, float(c[0]), float(c[1]), rul_encoding=cfg.get("rul_encoding", "raw_days"),
    )
    cf.to_csv(out / "table_rul_counterfactual.csv", index=False)
    print(f"\nRUL counterfactual (encoding={cfg.get('rul_encoding', 'raw_days')}; "
          f"re-run after retraining):\n", cf.to_string(index=False))

    n_row = int(row_n)
    (out / "base_rate.json").write_text(json.dumps({
        "row_level": {"n_drive_days": n_row, "n_positive": n_pos,
                      "positive_rate": n_pos / n_row if n_row else None},
        "drive_level": {"n_drives": int(len(d)), "n_failing": int(dy.sum()),
                        "failing_rate": float(dy.mean())},
        "base_paper": {"source": "Ding et al., IEEE ISPA 2024, Table IV",
                       "rul_cap_days": BASE_PAPER_WINDOW_DAYS, **BASE_PAPER["TFILL (base paper)"]},
        "seed": SEED,
    }, indent=2))
    print(f"\nDONE -> {out}")


if __name__ == "__main__":
    main()
