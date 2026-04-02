param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Get-BasePythonExe {
    $candidates = @(
        "C:\Users\suchida\AppData\Local\Programs\Python\Python312\python.exe",
        "C:\Users\suchida\anaconda3\python.exe"
    )

    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }

    throw "Base Python executable not found. Install Python first."
}

$basePython = Get-BasePythonExe
$venvDir = Join-Path $PSScriptRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if ($Recreate -and (Test-Path $venvDir)) {
    Write-Host "[INFO] Removing existing .venv"
    Remove-Item -Path $venvDir -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Creating .venv with: $basePython"
    & $basePython -m venv $venvDir
}
else {
    Write-Host "[INFO] Reusing existing .venv"
}

Write-Host "[INFO] Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Host "[INFO] Installing requirements.txt"
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

Write-Host "[OK] .venv is ready: $venvPython"
Write-Host "[INFO] Next steps:"
Write-Host "  .\run_connectivity_check.ps1"
Write-Host "  .\run_bt_sweep_inproc.ps1"
