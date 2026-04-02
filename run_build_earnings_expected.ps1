param(
    [string]$TargetYear = "",
    [string]$Tickers = "",
    [string]$MinSamples = "2",
    [string]$PreDays = "3",
    [string]$PostDays = "1"
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

if ($TargetYear -ne "") { $env:EARNINGS_EXPECTED_YEAR = $TargetYear }
if ($Tickers -ne "") { $env:EARNINGS_EXPECTED_TICKERS = $Tickers }
$env:EARNINGS_MIN_SAMPLES = $MinSamples
$env:EARNINGS_WINDOW_PRE_DAYS = $PreDays
$env:EARNINGS_WINDOW_POST_DAYS = $PostDays

& $pythonExe (Join-Path $PSScriptRoot 'build_earnings_expected.py')
