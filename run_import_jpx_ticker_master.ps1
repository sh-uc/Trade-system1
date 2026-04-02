param(
    [string]$XlsPath,
    [string]$OutCsv = "docs/jpx_ticker_master_prime_standard_latest.csv",
    [string]$SyncTickers = "1"
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

if (-not $XlsPath) {
    throw "XlsPath is required."
}

$pythonExe = Get-PythonExe
$env:JPX_TICKER_XLS = $XlsPath
$env:JPX_TICKER_MASTER_CSV = $OutCsv
$env:JPX_SYNC_TICKERS = $SyncTickers

& $pythonExe (Join-Path $PSScriptRoot 'import_jpx_ticker_master_xls.py')
