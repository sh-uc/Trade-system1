param(
    [string]$StrategyName = "swing_v1_candidate9",
    [string]$AsOf = "",
    [string]$PositionSimulationName = "",
    [string]$ReportDir = "docs/daily_reports",
    [string]$EarningsBlockDays = "2",
    [string]$EarningsBlockLabels = "FY",
    [string]$ExDividendBlockDays = "1",
    [string]$UpcomingEventDays = "7"
)

$ErrorActionPreference = "Stop"

function Get-PythonExe {
    $candidates = @(
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        "C:\Users\suchida\AppData\Local\Programs\Python\Python312\python.exe",
        "C:\Users\suchida\anaconda3\python.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    throw "Python executable not found. Create .venv or install Python."
}

$pythonExe = Get-PythonExe
Write-Host "[INFO] Using Python: $pythonExe"

$env:STRATEGY_NAME = $StrategyName
if ($AsOf -ne "") { $env:MONITOR_DATE = $AsOf }
if ($PositionSimulationName -ne "") { $env:POSITION_SIMULATION_NAME = $PositionSimulationName }
$env:REPORT_DIR = $ReportDir
$env:EARNINGS_BLOCK_DAYS = $EarningsBlockDays
$env:EARNINGS_BLOCK_LABELS = $EarningsBlockLabels
$env:EX_DIVIDEND_BLOCK_DAYS = $ExDividendBlockDays
$env:UPCOMING_EVENT_DAYS = $UpcomingEventDays

& $pythonExe (Join-Path $PSScriptRoot 'daily_paper_monitor.py')
