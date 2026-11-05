"""
17 -- Matched-protocol ablation comparison (arm A vs arm B)

Implements the statistics fixed in
`results_ablation/PREREGISTRATION_infra_ablation.md` section 6:

  * primary metric: row-level AUPRC over the FULL test split;
  * paired drive-clustered bootstrap on dAUPRC = AUPRC(A) - AUPRC(B),
    B=1000, 95% percentile interval, resampling PHYSICAL DRIVES;
  * per-seed sign consistency (>=4 of 5), from `17b_per_seed_auprc.py`;
  * cohort diagnostic: share of test rows in vaults never seen in training,
    and whether the gap concentrates there.

Why the bootstrap is written this way
-------------------------------------
Resampling 352k drives and materialising the concatenated rows is ~145M rows
per replicate -- impossible 1000 times.  But rows keep their relative order by
score, so a drive's multiplicity m_d can be applied as a *weight* on each of
its rows:

    AP = (1/P_w) * sum over positive rows i of  sum_{t=1..m_i} TP(t)/N(t)

where TP(t)/N(t) is the precision of the t-th copy of that row and P_w is the
weighted positive count.  The inner sum is exact (multiplicities are small
integers), not an approximation.  Sorting is done once; each replicate costs
two cumulative sums.

Usage
-----
    python 17_ablation_compare.py \
        --a_dir results_ablation/without_infra \
        --b_dir results_ablation/with_infra \
        --train data/processed/train_resampled.parquet \
        --test  data/processed/features_test.parquet \
        --boot 1000 \
        --out_prefix results_ablation/ablation
"""
import argparse
import gc
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
DELTA_MARGIN = 0.003


# ---------------------------------------------------------------- cache build

def build_cache(a_csv: Path, b_csv: Path, test_parquet: Path, out: Path) -> None:
    """Join the two arms' predictions on (serial_number, date) and attach a
    dense drive id and a dense vault id.  Written as parquet so the analysis
    can be re-run without repeating several 145M-row joins."""
    if out.exists():
        print(f"[CACHE] reusing {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return

    tmp = out.parent / "_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=3")
    con.execute("SET memory_limit='5GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    # Typed CSV reads: inference would scan each 5 GB file twice, and the
    # 144.7M-row join is already the memory-critical step.
    schema = ("{'serial_number':'VARCHAR','date':'DATE',"
              "'lgbm_prob':'FLOAT','failure_within_14d':'TINYINT'}")
    a_src = (f"read_csv('{a_csv.as_posix()}', header=true, columns={schema})")
    b_src = (f"read_csv('{b_csv.as_posix()}', header=true, columns={schema})")

    print("[CACHE] vault dimension ...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dim_vault AS
        SELECT vault_id,
               (row_number() OVER (ORDER BY vault_id) - 1)::SMALLINT AS vid
        FROM (SELECT DISTINCT vault_id FROM read_parquet('{test_parquet.as_posix()}'))
    """)
    dim_vault = con.execute("SELECT vault_id, vid FROM dim_vault").df()
    dim_vault.to_csv(out.with_name(out.stem + "_vault_codes.csv"), index=False)
    print(f"[CACHE] {len(dim_vault):,} vaults")

    print("[CACHE] drive dimension ...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dim_drive AS
        SELECT serial_number,
               (row_number() OVER (ORDER BY serial_number) - 1)::INTEGER AS drive_id
        FROM (SELECT DISTINCT serial_number FROM {a_src})
    """)
    n_drives = con.execute("SELECT count(*) FROM dim_drive").fetchone()[0]
    dim_drive = con.execute("SELECT serial_number, drive_id FROM dim_drive").df()
    dim_drive.to_csv(out.with_name(out.stem + "_drive_codes.csv"), index=False)
    print(f"[CACHE] {n_drives:,} drives")

    print("[CACHE] drive -> vault (vault is a property of the drive, so this "
          "joins on serial only) ...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE drive_vault AS
        SELECT t.serial_number, any_value(t.vault_id) AS vault_id
        FROM read_parquet('{test_parquet.as_posix()}') t
        GROUP BY t.serial_number
    """)

    print("[CACHE] joining arms and writing (this is the expensive step) ...")
    t0 = time.time()
    # No ORDER BY here.  The pre-registration's ordering requirement exists so
    # that an *evaluation* cannot depend on row order; AUPRC sorts by score and
    # ties are grouped, so it is order-invariant.  Sorting 145M rows inside the
    # COPY is what exhausted the 7 GB budget on the first attempt.
    con.execute(f"""
        COPY (
            SELECT d.drive_id,
                   a.lgbm_prob AS pA,
                   b.lgbm_prob AS pB,
                   a.failure_within_14d AS y,
                   dv.vid
            FROM {a_src} a
            JOIN {b_src} b
              ON a.serial_number = b.serial_number AND a.date = b.date
            JOIN dim_drive d
              ON a.serial_number = d.serial_number
            JOIN drive_vault t
              ON a.serial_number = t.serial_number
            LEFT JOIN dim_vault dv
              ON t.vault_id = dv.vault_id
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    print(f"[CACHE] written in {(time.time() - t0) / 60:.1f} min -> {out}")

    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"[CACHE] {n:,} rows")

    # Built-in join check: the test split has a known size and positive count
    # (from each arm's confusion matrix).  A duplicate or dropped join row shows
    # up here instead of silently shifting every downstream metric.
    ref = pd.read_csv(Path("archive/model1/results_ablation") / a_csv.parent.name / "lgbm_metrics.csv")
    expect_pos = int(ref["tp"].iloc[0] + ref["fn"].iloc[0])
    got_pos = con.execute(
        f"SELECT sum(y) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"[CACHE] positives {int(got_pos):,} (expected {expect_pos:,})")
    if int(got_pos) != expect_pos:
        raise RuntimeError(f"join lost or duplicated positives: {got_pos} != {expect_pos}")


# ---------------------------------------------------------------- statistics

def weighted_ap(y_sorted: np.ndarray, score_sorted: np.ndarray,
                drive_sorted: np.ndarray, mult: np.ndarray) -> float:
    """Exact AP under drive multiplicities, on rows already in descending score
    order.

    sklearn computes AP over distinct score thresholds: for each threshold
    group g, precision is TP_after(g)/N_after(g) and the recall jump is
    positives(g)/P.  Rows tied on score therefore have to be collapsed into one
    group -- an earlier version applied the multiplicity row-by-row and
    disagreed with sklearn by 3e-4 whenever ties existed.

    A drive drawn m times contributes each of its rows m times, so the group
    sums are just the group sums of the per-row multiplicity.
    """
    w = mult[drive_sorted]
    nz = w > 0
    if not nz.any():
        return float("nan")
    s = score_sorted[nz]
    ys = y_sorted[nz]
    ws = w[nz].astype(np.int32, copy=False)

    if len(s) > 1:
        starts = np.concatenate(([0], np.flatnonzero(s[1:] != s[:-1]) + 1))
    else:
        starts = np.zeros(1, dtype=np.int64)

    wg = np.add.reduceat(ws, starts).astype(np.float64)
    pg = np.add.reduceat(ws * ys, starts).astype(np.float64)

    total_pos = float(pg.sum())
    if total_pos <= 0:
        return float("nan")

    tp = np.cumsum(pg)
    wn = np.cumsum(wg)
    return float(np.sum((pg / total_pos) * (tp / wn)))


def bootstrap_delta(pA, pB, y, drive_id, n_boot: int):
    """Paired drive-clustered bootstrap on dAUPRC."""
    n_drives = int(drive_id.max()) + 1
    print(f"[BOOT] sorting 2 x {len(y):,} rows by score ...")
    orderA = np.argsort(-pA, kind="stable")
    orderB = np.argsort(-pB, kind="stable")
    yA, drA, sA = y[orderA], drive_id[orderA], pA[orderA]
    yB, drB, sB = y[orderB], drive_id[orderB], pB[orderB]
    # sort keys no longer needed: only the sorted views are reused
    del pA, pB, orderA, orderB, drive_id
    gc.collect()

    rng = np.random.default_rng(SEED)
    gaps = np.empty(n_boot, dtype=np.float64)
    done = 0
    t0 = time.time()
    for b in range(n_boot):
        pick = rng.integers(0, n_drives, n_drives)
        mult = np.bincount(pick, minlength=n_drives).astype(np.int32)
        ap_a = weighted_ap(yA, sA, drA, mult)
        ap_b = weighted_ap(yB, sB, drB, mult)
        if np.isnan(ap_a) or np.isnan(ap_b):
            continue
        gaps[done] = ap_a - ap_b
        done += 1
        if (b + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[BOOT] {b + 1}/{n_boot}  "
                  f"({el / (b + 1):.2f}s/rep, eta {(n_boot - b - 1) * el / (b + 1) / 60:.1f} min)")
    return gaps[:done]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_dir", default="archive/model1/results_ablation/without_infra")
    ap.add_argument("--b_dir", default="archive/model1/results_ablation/with_infra")
    ap.add_argument("--train", default="data/processed/train_resampled.parquet")
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out_prefix", default="archive/model1/results_ablation/ablation")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--reuse_bootstrap", action="store_true",
                    help="reuse saved replicates instead of recomputing them "
                         "(use when only the per-seed/per-cohort part changed)")
    args = ap.parse_args()

    a_dir, b_dir = Path(args.a_dir), Path(args.b_dir)
    out_prefix = Path(args.out_prefix)
    cache = out_prefix.with_name("_ablation_joined.parquet")

    a_csv = a_dir / "lgbm_test_probs.csv"
    b_csv = b_dir / "lgbm_test_probs.csv"
    for p in (a_csv, b_csv):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- arm not finished")

    if args.rebuild:
        cache.unlink(missing_ok=True)
    build_cache(a_csv, b_csv, Path(args.test), cache)

    print("[LOAD] reading joined predictions ...")
    tbl = pq.read_table(cache, columns=["pA", "pB", "y", "drive_id", "vid"])
    pA = tbl["pA"].to_numpy()
    pB = tbl["pB"].to_numpy()
    y = tbl["y"].to_numpy()
    drive_id = tbl["drive_id"].to_numpy()
    vid = tbl["vid"].fill_null(-1).to_numpy()
    del tbl
    gc.collect()

    n = len(y)
    prev = float(y.mean())
    print(f"[LOAD] {n:,} rows, {int(y.sum()):,} positives, prevalence {prev:.6%}")
    print(f"[LOAD] {int(drive_id.max()) + 1:,} drives, "
          f"{int(np.unique(vid).size):,} vault codes")

    apA = float(average_precision_score(y, pA))
    apB = float(average_precision_score(y, pB))
    rocA = float(roc_auc_score(y, pA))
    rocB = float(roc_auc_score(y, pB))
    delta = apA - apB
    print(f"\n[POINT] arm A (212 feat) AUPRC {apA:.10f}  ROC {rocA:.10f}")
    print(f"[POINT] arm B (217 feat) AUPRC {apB:.10f}  ROC {rocB:.10f}")
    print(f"[POINT] delta (A - B)    {delta:+.10f}  "
          f"({100 * delta / apB:+.2f}% relative)")

    boot_csv = out_prefix.with_name("ablation_bootstrap.csv")
    if args.reuse_bootstrap and boot_csv.exists():
        gaps = pd.read_csv(boot_csv)["delta"].to_numpy()
        print(f"[BOOT] reusing {len(gaps)} saved replicates from {boot_csv}")
    else:
        gaps = bootstrap_delta(pA, pB, y, drive_id, args.boot)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    wins_pct = 100.0 * float((gaps > 0).mean())
    print(f"\n[BOOT] delta {gaps.mean():+.10f}  95% CI [{lo:+.10f}, {hi:+.10f}]  "
          f"B={len(gaps)}  A>B in {wins_pct:.1f}% of resamples")

    # ---- per-seed -------------------------------------------------------
    seed_rows = []
    pa = Path("archive/model1/results_ablation") / f"per_seed_full_{a_dir.name}.csv"
    pb = Path("archive/model1/results_ablation") / f"per_seed_full_{b_dir.name}.csv"
    wins = 0
    n_seed_pairs = 0
    if pa.exists() and pb.exists():
        da = pd.read_csv(pa)
        db = pd.read_csv(pb)
        da = da[da["seed"] != "ensemble"].copy()
        db = db[db["seed"] != "ensemble"].copy()
        m = da.merge(db, on="seed", suffixes=("_A", "_B"))
        m["delta"] = m["auprc_A"] - m["auprc_B"]
        wins = int((m["delta"] > 0).sum())
        n_seed_pairs = int(len(m))
        print("\n[PER-SEED] full test split")
        print(m[["seed", "auprc_A", "auprc_B", "delta"]].to_string(index=False))
        print(f"[PER-SEED] A wins {wins}/{len(m)}")
        m.to_csv(out_prefix.with_name("ablation_per_seed.csv"), index=False)
    else:
        print(f"\n[PER-SEED] missing per-seed file for an arm; "
              f"run 17c_per_seed_full.py on both (expected {pa.name}, {pb.name})")
        wins = -1

    # ---- cohort ---------------------------------------------------------
    # Two distinct novelty channels.  An infrastructure identifier can help by
    # memorising the site (vault) OR by memorising the individual drive's
    # history; these are different mechanisms and are measured separately.
    con = duckdb.connect()
    con.execute("SET threads=4")
    train_vaults = set(con.execute(
        f"SELECT DISTINCT vault_id FROM read_parquet('{Path(args.train).as_posix()}')"
    ).df()["vault_id"].astype(str))
    train_serials = set(con.execute(
        f"SELECT DISTINCT serial_number FROM read_parquet('{Path(args.train).as_posix()}')"
    ).df()["serial_number"].astype(str))

    codes = pd.read_csv(cache.with_name(cache.stem + "_vault_codes.csv"))
    codes["vault_id"] = codes["vault_id"].astype(str)
    train_vids = codes.loc[codes["vault_id"].isin(train_vaults), "vid"].to_numpy()
    unseen_vault = ~np.isin(vid, train_vids)

    dcodes = pd.read_csv(cache.with_name(cache.stem + "_drive_codes.csv"))
    drive_seen = np.zeros(int(dcodes["drive_id"].max()) + 1, dtype=bool)
    seen_ids = dcodes.loc[dcodes["serial_number"].astype(str).isin(train_serials),
                          "drive_id"].to_numpy()
    drive_seen[seen_ids] = True
    unseen_drive = ~drive_seen[drive_id]

    print(f"\n[COHORT] vault novelty: {100.0 * unseen_vault.mean():.3f}% of test rows "
          f"({int(unseen_vault.sum()):,}) sit in vaults never seen in training")
    print(f"[COHORT] drive novelty: {100.0 * unseen_drive.mean():.3f}% of test rows "
          f"({int(unseen_drive.sum()):,}) belong to drives never seen in training")

    cohort = []
    splits = [
        ("vault_seen", "vault", ~unseen_vault),
        ("vault_unseen", "vault", unseen_vault),
        ("drive_seen", "drive", ~unseen_drive),
        ("drive_unseen", "drive", unseen_drive),
    ]
    for name, dim, m_ in splits:
        if m_.sum() == 0:
            continue
        yy = y[m_]
        if yy.sum() == 0:
            print(f"[COHORT] {name}: no positives, skipped")
            continue
        a_ = float(average_precision_score(yy, pA[m_]))
        b_ = float(average_precision_score(yy, pB[m_]))
        cohort.append({
            "cohort": name, "dimension": dim, "rows": int(m_.sum()),
            "positives": int(yy.sum()), "prevalence": float(yy.mean()),
            "auprc_A": a_, "auprc_B": b_, "delta": a_ - b_,
            "roc_A": float(roc_auc_score(yy, pA[m_])),
            "roc_B": float(roc_auc_score(yy, pB[m_])),
        })
        print(f"[COHORT] {name:<13} rows {int(m_.sum()):>10,}  pos {int(yy.sum()):>6,}  "
              f"AUPRC A {a_:.6f} B {b_:.6f} delta {a_ - b_:+.6f}")
    pd.DataFrame(cohort).to_csv(out_prefix.with_name("cohort_test.csv"), index=False)

    # ---- decision -------------------------------------------------------
    if hi < 0:
        verdict = "REVERSAL"
        note = "identifiers help; restore them and retract the leakage framing"
    elif lo > 0 and wins >= 4:
        verdict = "SUPERIORITY"
        note = "removing identifiers improves ranking"
    elif lo > -DELTA_MARGIN and hi < DELTA_MARGIN and wins < 4:
        verdict = "EQUIVALENCE"
        note = "identifiers are immaterial; keep 212 for portability"
    elif lo > -DELTA_MARGIN:
        verdict = "NON-INFERIORITY"
        note = "removal does not materially harm; keep 212"
    else:
        verdict = "INCONCLUSIVE"
        note = "interval too wide to decide"
    print(f"\n[VERDICT] {verdict} -- {note}")

    summary = [{
        "arm_A": a_dir.name, "arm_B": b_dir.name,
        "n_rows": n, "n_positives": int(y.sum()), "prevalence": prev,
        "auprc_A": apA, "auprc_B": apB, "delta": delta,
        "roc_A": rocA, "roc_B": rocB,
        "delta_pct": 100 * delta / apB,
        "boot_mean": float(gaps.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
        "a_wins_pct": wins_pct, "n_boot": int(len(gaps)),
        "seed_wins": wins, "seed_total": n_seed_pairs,
        "verdict": verdict,
    }]
    pd.DataFrame(summary).to_csv(out_prefix.with_name("ablation_summary.csv"), index=False)
    pd.DataFrame({"delta": gaps}).to_csv(
        out_prefix.with_name("ablation_bootstrap.csv"), index=False)
    print(f"[SAVE] {out_prefix.with_name('ablation_summary.csv')}")
    print(f"[SAVE] {out_prefix.with_name('ablation_bootstrap.csv')}")


if __name__ == "__main__":
    main()
