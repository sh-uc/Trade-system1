param(
    [string]$Ticker = "3778.T"
)

$ErrorActionPreference = "Stop"

function Get-PythonExe {
    $candidates = @(
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        "C:\Users\suchida\AppData\Local\Programs\Python\Python312\python.exe",
        "C:\Users\suchida\anaconda3\python.exe"
    )

    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }

    throw "Python executable not found. Create .venv or install Python."
}

$pythonExe = Get-PythonExe
Write-Host "[INFO] Using Python: $pythonExe"
$env:CHECK_TICKER = $Ticker
& $pythonExe (Join-Path $PSScriptRoot 'connectivity_check.py')
