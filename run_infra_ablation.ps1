# Matched-pair infrastructure-identifier ablation.
# Protocol: results_ablation/PREREGISTRATION_infra_ablation.md
#
# Fixes vs the first attempt (2026-10-23 08:23):
#   1. --train was data/processed/features_train.parquet (484,139,844 rows, 21.7 GB
#      on disk). The model trains on data/processed/train_resampled.parquet
#      (3,440,506 rows). The wrong file caused the arm to die with exit 1.
#   2. Log() used Write-Output inside a function, so Run-Arm returned an array of
#      strings plus its boolean and `-not $ok` never fired. Failures now throw.
#
# Runs sequentially (never concurrently) to stay inside the 15.7 GB RAM budget.
# Shipped models/ and results/ are never written.

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$py      = ".\.venv\Scripts\python.exe"
$log     = "results_ablation\run.log"
$train   = "data/processed/train_resampled.parquet"
$val     = "data/processed/features_val.parquet"
$test    = "data/processed/features_test.parquet"
$cost    = "data/processed/cost_matrix.json"
$seeds   = "42,7,13,99,23"

New-Item -ItemType Directory -Force -Path results_ablation | Out-Null

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line
}

# ---- preflight: refuse to start on the wrong training file -------------------
Log "PREFLIGHT"
$env:ABL_TRAIN  = $train
$env:ABL_VAL    = $val
$env:ABL_TEST   = $test
$rows = & $py -c "import os, pyarrow.parquet as pq; print(pq.ParquetFile(os.environ['ABL_TRAIN']).metadata.num_rows)"
if ($LASTEXITCODE -ne 0) { Log "PREFLIGHT FAILED: cannot read $train"; exit 1 }
$rows = [int64]$rows
Log ("train rows = {0:N0}" -f $rows)
if ($rows -gt 5000000) {
    Log "ABORT: $train has $rows rows; expected the 3.4M-row resampled set."
    Log "The 484M-row features_train.parquet will exhaust RAM. Refusing to run."
    exit 1
}
foreach ($p in @($train, $val, $test, $cost)) {
    if (-not (Test-Path $p)) { Log "ABORT: missing $p"; exit 1 }
}
Log "preflight OK"

function Run-Arm([string]$name, [string[]]$extra) {
    $mdir = "models_ablation\$name"
    $rdir = "results_ablation\$name"
    New-Item -ItemType Directory -Force -Path $mdir, $rdir | Out-Null

    # A finished arm is never re-run. Re-running predict would cost ~4 hours and
    # would overwrite an identical result, so treat existing metrics as final.
    if ((Test-Path "$rdir\lgbm_metrics.csv") -and
        (Test-Path "$rdir\lgbm_test_probs.csv") -and
        ((Get-Item "$rdir\lgbm_test_probs.csv").Length -gt 1GB)) {
        Log "=== ARM $name : COMPLETE (metrics + test probs on disk) -- skipping ==="
        return
    }

    # Resume-aware: if this arm already finished training, do not retrain --
    # Optuna draws different hyper-parameters each run, so retraining would
    # silently discard the arm's recorded tuning result.
    $trained = (Test-Path "$mdir\lgbm_ensemble_seeds.json") -and
               (Test-Path "$mdir\lgbm_best_params.json") -and
               ((Get-ChildItem "$mdir\lgbm_model_seed*.txt" -ErrorAction SilentlyContinue).Count -ge 5)
    if ($trained) {
        Log "=== ARM $name : TRAIN already complete (models on disk) -- reusing ==="
    } else {
        Log "=== ARM $name : TRAIN ==="
        & $py 06_train_lightgbm.py --mode train `
            --train $train --val $val --target failure_within_14d `
            --n_trials 30 --seeds $seeds @extra --out_dir $mdir $rdir
        if ($LASTEXITCODE -ne 0) { throw "TRAIN FAILED for $name (exit $LASTEXITCODE)" }
    }

    Log "=== ARM $name : PREDICT ==="
    & $py 06_train_lightgbm.py --mode predict `
        --val $val --test $test --cost_matrix $cost `
        --target failure_within_14d --seeds $seeds --ensemble_ckpt `
        @extra --out_dir $mdir $rdir
    if ($LASTEXITCODE -ne 0) { throw "PREDICT FAILED for $name (exit $LASTEXITCODE)" }

    Log "=== ARM $name : DONE ==="
}

Log "ABLATION START"
try {
    Run-Arm "without_infra" @()
    Run-Arm "with_infra" @("--keep_infra_features")
} catch {
    Log "ABLATION FAILED: $($_.Exception.Message)"
    exit 1
}
Log "ABLATION COMPLETE -- run 17_ablation_compare.py next"
