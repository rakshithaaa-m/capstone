# RESUME HERE — 14 Oct 2026

## State at pause

Everything is committed. No heavy processes are running. Drive `D:` was
connected and healthy at pause (931.5 GB, Disk 1, status OK).

---

## DONE TODAY (all committed)

### 1. Stacking study — the fusion question is now settled on evidence

`16_stacking_study.py` (commit `3ac0053`), outputs in `results/stacking_study*.csv`.

Tested five combiners on out-of-fold validation predictions, drive-level
"ever-fails" label, 1,353 positives on the selection half:

| combiner | AUPRC | vs LightGBM |
|---|---|---|
| **LightGBM alone** | **0.60526** | — |
| logistic [prob, risk] | 0.58878 | −0.01648 |
| GBDT + engineered | 0.57488 | −0.03038 |
| GBDT [prob, risk] | 0.54075 | −0.06451 |
| GBDT prob-only (control) | 0.53612 | −0.06914 |

Paired drive-clustered bootstrap, B=400:
**gap −0.01648, 95% CI [−0.02457, −0.00915], wins 0.0% of resamples.**

Because the prob-only *control* also lost, the combiner comparison alone is
confounded (a GBDT on two derived columns is simply a weaker learner than the
212-feature LightGBM). So a model-free conditional test was added: within
deciles of LightGBM probability, AUROC of the RUL-derived risk score is
**0.4928 median (4/10 bands above 0.5)**, 0.5264 marginally.

> RUL carries no information beyond what the classifier already extracts.
> `lgbm_only` at Stage 8 is correct, now on independent evidence.

### 2. Two methodology bugs caught before they reached the paper

- **Label too small.** `target` on a drive's last observed day is almost never
  1, leaving 69 positives and a drive-level AUPRC CI spanning a 5× range.
  A first run appeared to show a *doubling* from the GBDT stacker (+0.0199);
  on re-run it reversed. Root cause: DuckDB does not guarantee row order, so an
  unstable input order reshuffled the fit/select split. Fixed by labelling with
  `MAX(target)` over all rows (2,770 positives) and sorting by serial first.
  **Never report a <1% AUPRC gap here without the drive-clustered bootstrap.**
- **Self-comparison.** The bootstrap originally read column 0 of the feature
  matrix — which is `prob` for every spec — so it compared LightGBM with
  itself. Now bootstraps the winner's actual stored predictions.

### 3. Downstream chain re-run onto the corrected RUL

Stages 9→12 re-run and committed-clean (all Oct 14):

- **Stage 9** — cutoffs unchanged (HIGH 0.76406634, MEDIUM 0.06907166);
  tiers unchanged at HIGH 6,282 / MEDIUM 26,512 / LOW 319,589 drives. Expected:
  `lgbm_only` means the hybrid score *is* the LightGBM probability.
  Test tier report: HIGH captures **52.30%** of all failures from **0.59%** of rows.
- **Stage 10** — re-ran with `--skip_phases 2a,2b,3`. Those three phases are
  LightGBM-only ablations, mathematically independent of the RUL fix, and each
  retrains + streams 145M rows (~45 min each, wasted). Outputs from 30 Sep/1 Oct
  are still valid — `lgbm_feature_cols.json` unchanged since 29 Sep.
  Ran in 9 min instead of 90+.
- **Stage 11** — SHAP unchanged and still infra-free: `smart_1_raw` 0.2517,
  `error_total` 0.2463, `smart_18_normalized` 0.2132. No `vault_id`.
- **Stage 12** — 32,794 mitigation records, full coverage.
- **Serving DB rebuilt** → `{'ok': True, 'stale': False, 'count': 352383}`.

### 4. Datacenter holdout — found and fixed three defects

`15_datacenter_holdout.py` (commit `6c4aaa0`). **It was not measuring what it
claimed.** The original `train_sub = train_df` meant the model was fitted on the
held-out datacenter's own drives — so "held-out" AUPRC said nothing about
portability, while the docstring claimed "trains on the rest". Also had no
control fold (any drop was ambiguous between site shift and less training
data) and `USING SAMPLE` with no seed.

Fixed: true holdout from training, a size-matched random control fold per
datacenter fold, seeded sampling, map built from all three splits (488,142
drives vs 352,383 from test alone), and hyperparameters reused from
`models/lgbm_best_params.json` instead of a fresh 10-trial Optuna search per
fold (>60 min/fold → should be minutes).

---

## TODO TOMORROW — in this order

### 1. Finish the datacenter holdout (the only unrun item)

The fix is written and compiles, but the validation run was cut short. Launch:

```powershell
.\.venv\Scripts\python.exe -u 15_datacenter_holdout.py --out results/datacenter_holdout.csv
```

If it is still slow, smoke-test the fold logic first with
`--n_holdout 1 --max_test_rows 50000`.

**What to check in the output** (`results/datacenter_holdout.csv`): the `kind`
column separates `datacenter:` from `control:` folds. The number that matters is
median control AUPRC − median held-out AUPRC, printed as
"Datacenter-shift cost". Near zero means the model relies on drive health, not
site identity, and the claim is defensible.

### 2. Regenerate the paper tables (they are from 1 Oct, pre-correction)

```powershell
.\.venv\Scripts\python.exe -u 14b_paper_evaluation.py --results_dir results --boot 200 --boot-row 25
```

~20 min. Writes `results/paper_tables/` (7 tables, all stale).

### 3. Re-run the baseline comparison with current isotonic calibration

```powershell
.\.venv\Scripts\python.exe -u fix2_baseline_comparison.py
```

Current: LR 0.0181, XGBoost 0.0020, LightGBM 0.0839 (4.6× the next best).

### 4. Add the stacking study to the paper

`results/stacking_study.csv`, `_bootstrap.csv`, `_conditional.csv`. Frame it as
the quantified negative result on RUL — it is the strongest evidence in the
project, not a failure. The conditional table (`_conditional.csv`) is the
model-free argument and is the one to put in the paper.

---

## Things deliberately NOT done (do not re-litigate tomorrow)

- **Do not force a fusion.** Five combiners lost. Report it.
- **Do not re-run `ablation_infra_identifiers.csv`.** The shipped config already
  excludes the identifiers; the file is annotated rather than re-run.
- **Do not implement TFILL-style time-mixing.** Weeks of work, and only 2
  ST4000DM000 drives / 0 failures in the test split, so a matched comparison is
  impossible.
- **The 99.34 MAE baseline is unrecoverable** (weights were overwritten before
  snapshot — my error). State it, do not try to reconstruct it.

## Honest caveats to carry into any write-up

1. RUL MAE (~96 d) is ~35× worse than the published TFILL result (5.04) — on a
   **non-matched dataset**; their evaluation drive model has 2 drives and 0
   failures in our test split. State this before a reviewer finds it.
2. Zero FP and zero FN is **mathematically impossible** at 0.0468% prevalence.
3. Fusion losing is the **correct, reported outcome**, not a defect.
