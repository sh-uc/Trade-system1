param(
    [string]$SimulationName = "paper_main",
    [string]$StrategyName = "swing_v1",
    [string]$Start = "2026-01-01",
    [string]$End = "",
    [string]$Tickers = "",
    [string]$Capital = "3000000",
    [string]$Reset = "0",
    [string]$EarningsBlockDays = "",
    [string]$EarningsForceCloseDays = "",
    [string]$EarningsBlockLabels = "",
    [string]$EarningsForceCloseLabels = "",
    [string]$ExDividendBlockDays = "",
    [string]$ExDividendForceCloseDays = "",
    [string]$MaxTotalExposure = ""
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

$env:SIMULATION_NAME = $SimulationName
$env:STRATEGY_NAME = $StrategyName
$env:SIM_START = $Start
if ($End -ne "") { $env:SIM_END = $End }
if ($Tickers -ne "") { $env:SIM_TICKERS = $Tickers }
$env:SIM_CAPITAL = $Capital
$env:SIM_RESET = $Reset
if ($EarningsBlockDays -ne "") { $env:EARNINGS_BLOCK_DAYS = $EarningsBlockDays }
if ($EarningsForceCloseDays -ne "") { $env:EARNINGS_FORCE_CLOSE_DAYS = $EarningsForceCloseDays }
if ($EarningsBlockLabels -ne "") { $env:EARNINGS_BLOCK_LABELS = $EarningsBlockLabels }
if ($EarningsForceCloseLabels -ne "") { $env:EARNINGS_FORCE_CLOSE_LABELS = $EarningsForceCloseLabels }
if ($ExDividendBlockDays -ne "") { $env:EX_DIVIDEND_BLOCK_DAYS = $ExDividendBlockDays }
if ($ExDividendForceCloseDays -ne "") { $env:EX_DIVIDEND_FORCE_CLOSE_DAYS = $ExDividendForceCloseDays }
if ($MaxTotalExposure -ne "") { $env:MAX_TOTAL_EXPOSURE = $MaxTotalExposure }

& $pythonExe (Join-Path $PSScriptRoot 'simulate_live.py')
