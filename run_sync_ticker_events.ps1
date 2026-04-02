param(
    [string]$Tickers = ""
)

$ErrorActionPreference = "Stop"

function Get-PythonExe {
    $candidates = @(
        (Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"),
        "C:\\Users\\suchida\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
        "C:\\Users\\suchida\\anaconda3\\python.exe"
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

if ($Tickers -ne "") {
    $env:EVENT_TICKERS = $Tickers
}

& $pythonExe (Join-Path $PSScriptRoot 'sync_ticker_events.py')
