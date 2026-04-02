param(
    [string]$TraderSimulationName = "paper_candidate9_daily",
    [string]$StrategyName = "swing_v1_candidate9",
    [string]$TraderDate = "",
    [string]$ReportDir = "docs/daily_reports",
    [string]$Capital = "3000000",
    [string]$MaxOpenPositions = "5",
    [string]$MaxNewPositionsPerDay = "2",
    [string]$MaxTotalExposure = "",
    [string]$EarningsBlockDays = "2",
    [string]$EarningsBlockLabels = "FY",
    [string]$EarningsForceCloseDays = "0",
    [string]$ExDividendBlockDays = "1",
    [string]$ExDividendForceCloseDays = "0",
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

$env:TRADER_SIMULATION_NAME = $TraderSimulationName
$env:STRATEGY_NAME = $StrategyName
if ($TraderDate -ne "") { $env:TRADER_DATE = $TraderDate }
$env:REPORT_DIR = $ReportDir
$env:SIM_CAPITAL = $Capital
$env:MAX_OPEN_POSITIONS = $MaxOpenPositions
$env:MAX_NEW_POSITIONS_PER_DAY = $MaxNewPositionsPerDay
if ($MaxTotalExposure -ne "") { $env:MAX_TOTAL_EXPOSURE = $MaxTotalExposure }
$env:EARNINGS_BLOCK_DAYS = $EarningsBlockDays
$env:EARNINGS_BLOCK_LABELS = $EarningsBlockLabels
$env:EARNINGS_FORCE_CLOSE_DAYS = $EarningsForceCloseDays
$env:EX_DIVIDEND_BLOCK_DAYS = $ExDividendBlockDays
$env:EX_DIVIDEND_FORCE_CLOSE_DAYS = $ExDividendForceCloseDays
$env:UPCOMING_EVENT_DAYS = $UpcomingEventDays

& $pythonExe (Join-Path $PSScriptRoot 'daily_paper_trader.py')
