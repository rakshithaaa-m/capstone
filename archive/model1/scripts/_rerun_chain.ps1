# Resume the downstream chain from Stage 10.
# Stage 9 already completed on the censoring-corrected LSTM.
# Phases 2a/2b/3 of Stage 10 are LightGBM-only ablations: they retrain the
# classifier and stream the full 145M-row test set, and they are mathematically
# independent of the LSTM/RUL fix, so re-running them is pure waste. Their
# outputs already exist from the 30 Sep / 1 Oct run against the same LightGBM
# model (lgbm_feature_cols.json unchanged since 29 Sep).
$ErrorActionPreference = "Continue"
$py = "D:\capstone\.venv\Scripts\python.exe"
Set-Location D:\capstone

function Step($name, $argList) {
    Write-Host ""
    Write-Host "=================== $name ==================="
    Write-Host "[$(Get-Date -Format HH:mm:ss)] starting $name"
    & $py @argList 2>&1 | Tee-Object -Variable out | Out-Null
    $code = $LASTEXITCODE
    $out | Select-Object -Last 14 | ForEach-Object { Write-Host "  $_" }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $name exit=$code"
    if ($code -ne 0) { Write-Host "!!! $name FAILED"; exit 1 }
}

Step "Stage 10 - evaluation" @(
    "-u","10_evaluation.py",
    "--train_resampled","data/processed/train_resampled.parquet",
    "--train_full","data/processed/features_train.parquet",
    "--val","data/processed/features_val.parquet",
    "--test","data/processed/features_test.parquet",
    "--cost_matrix","data/processed/cost_matrix.json",
    "--lgbm_metrics","results/lgbm_metrics.csv",
    "--lgbm_test_probs","results/lgbm_test_probs.csv",
    "--hybrid_metrics","results/hybrid_metrics.csv",
    "--hybrid_test_scores","results/hybrid_test_scores.csv",
    "--target","failure_within_14d",
    "--skip_phases","2a,2b,3",
    "--out_dir","results"
)

Step "Stage 11 - SHAP explainability" @(
    "-u","11_shap_explainability.py",
    "--test","data/processed/features_test.parquet",
    "--lgbm_model","models/lgbm_model.txt",
    "--lgbm_feature_cols","models/lgbm_feature_cols.json",
    "--risk_tiers","results/risk_tiers_test.csv",
    "--n_samples","3000",
    "--max_at_risk_rows","50000",
    "--out_dir","results"
)

Step "Stage 12 - mitigation mapping" @(
    "-u","12_mitigation_mapping.py",
    "--risk_tiers","results/risk_tiers_test.csv",
    "--shap_values","results/shap_values_at_risk.parquet",
    "--out","results/mitigation_mapping.json"
)

Write-Host ""
Write-Host "=== chain complete $(Get-Date -Format HH:mm:ss) ==="
