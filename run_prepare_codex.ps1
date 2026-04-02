$ErrorActionPreference = "Stop"

Write-Host "[INFO] Step 1/2: setup_venv.ps1"
& (Join-Path $PSScriptRoot 'setup_venv.ps1')

Write-Host "[INFO] Step 2/2: run_connectivity_check.ps1"
& (Join-Path $PSScriptRoot 'run_connectivity_check.ps1')
