# Back up the parts of the project that are slow or impossible to rebuild.
# Skipped (rebuildable): data/raw (re-download from Backblaze), data/processed/cleaned_by_quarter
# (Stage 2), retrain2026 fleet_population / fleet_pct / features_*_B (29_retrain_2026.py),
# derived CSVs (risk_tiers_test, LSTM / XGB-AFT prediction exports).
#   powershell -File backup_essentials.ps1 [-Dest C:\capstone_backup]
param([string]$Dest = "C:\capstone_backup")
$Src = $PSScriptRoot
$log = Join-Path $Dest "_backup_log.txt"
New-Item -ItemType Directory -Force $Dest | Out-Null

function Copy-Dir($rel, [string[]]$excludeDirs = @(), [string[]]$excludeFiles = @()) {
    $ra = @((Join-Path $Src $rel), (Join-Path $Dest $rel), "/E", "/COPY:DT", "/R:2", "/W:5", "/NP", "/NDL", "/LOG+:$log")
    if ($excludeDirs) { $ra += "/XD"; $ra += $excludeDirs | ForEach-Object { Join-Path $Src $_ } }
    if ($excludeFiles) { $ra += "/XF"; $ra += $excludeFiles }
    robocopy @ra | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed on $rel (exit $LASTEXITCODE)" }
    Write-Host "[OK] $rel"
}

Copy-Dir "data\processed" @("data\processed\cleaned_by_quarter", "data\processed\retrain2026\fleet_population",
                           "data\processed\retrain2026\fleet_pct", "data\processed\retrain2026\features_val_B",
                           "data\processed\retrain2026\features_test_B")
Copy-Dir "results" @() @("risk_tiers_test.csv", "lstm_full_test_preds.csv", "lstm_full_val_preds.csv")
Copy-Dir "results_fusion" @() @("pa_xgb_aft_test_preds.csv", "pa_xgb_aft_val_preds.csv")
foreach ($d in "models", "models_ablation", "models_fusion", "models_retrain2026", "results_ablation",
               "results_fresh2026", "results_retrain2026", "results_before_rerun") { Copy-Dir $d }
# deployment: small outputs and the agent evaluation; per-row scores rebuild with 31_deploy_2026.py
Copy-Dir "results_deploy2026" @("results_deploy2026\_work") @("risk_tiers_test.csv", "hybrid_test_scores.csv")
Write-Host "[DONE] $Dest"
