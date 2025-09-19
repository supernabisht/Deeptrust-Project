# Cleanup Script for DeepTrust Project

# Remove Python cache files
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force */__pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force */*/__pycache__ -ErrorAction SilentlyContinue

# Remove temporary files
Remove-Item -Recurse -Force *.pyc -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.pyo -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.pyd -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.py~ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .DS_Store -ErrorAction SilentlyContinue

# Remove old logs and cache
Remove-Item -Recurse -Force *.log -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.cache -ErrorAction SilentlyContinue

# Remove old virtual environments (keep the main one)
Remove-Item -Recurse -Force deeptrust_env -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force venv_final -ErrorAction SilentlyContinue

# Remove temporary directories
Remove-Item -Recurse -Force output_frames -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force lip_frames -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force temp -ErrorAction SilentlyContinue

# Remove old model versions (keeping the latest ones)
$keepModels = @(
    "models/optimized_deepfake_model.pkl",
    "models/optimized_feature_scaler.pkl",
    "models/optimized_label_encoder.pkl",
    "models/ultimate_deepfake_model.pkl"
)

# Get all model files
$allModels = Get-ChildItem -Path "models" -Filter "*.pkl" -Recurse -File | Select-Object -ExpandProperty FullName

# Remove old model files not in the keep list
foreach ($model in $allModels) {
    $keep = $false
    foreach ($keepModel in $keepModels) {
        if ($model -like "*$keepModel*") {
            $keep = $true
            break
        }
    }
    
    if (-not $keep) {
        Write-Host "Removing old model: $model"
        Remove-Item -Path $model -Force -ErrorAction SilentlyContinue
    }
}

# Remove empty directories
Get-ChildItem -Directory -Recurse | Where-Object { @(Get-ChildItem -Path $_.FullName -Recurse -Force | Where-Object { !$_.PSIsContainer }).Count -eq 0 } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Cleanup completed successfully!" -ForegroundColor Green
