param(
    [string]$Ticker = "3778.T",
    [string]$Start = "2024-01-01",
    [string]$MaxHoldDays = "5",
    [string]$VolSpikeM = "1.0,1.1,1.2",
    [string]$TakeProfitRr = "2.0,3.0,4.0",
    [string]$BreakEvenR = "",
    [string]$TrailingStartR = "",
    [string]$TrailingStopR = "",
    [string]$RiskPct = "0.004,0.006,0.008",
    [string]$StopPct = "",
    [string]$RsiMin = "",
    [string]$RsiMax = "80",
    [string]$MacdAtrK = "0.15",
    [string]$GapEntryMax = "",
    [string]$ExitOnReverse = "",
    [string]$UseIntradayResolution = "",
    [string]$IntradayInterval = "",
    [string]$IntradayTieBreak = "",
    [string]$SweepWorkers = "4",
    [string]$SaveBt = "1"
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

$env:SWEEP_TICKERS = $Ticker
$env:SWEEP_START = $Start
$env:MAX_HOLD_DAYS = $MaxHoldDays
$env:VOL_SPIKE_M = $VolSpikeM
$env:TAKE_PROFIT_RR = $TakeProfitRr
if ($BreakEvenR -ne "") {
    $env:BREAK_EVEN_R = $BreakEvenR
}
if ($TrailingStartR -ne "") {
    $env:TRAILING_START_R = $TrailingStartR
}
if ($TrailingStopR -ne "") {
    $env:TRAILING_STOP_R = $TrailingStopR
}
$env:RISK_PCT = $RiskPct
if ($StopPct -ne "") {
    $env:STOP_PCT = $StopPct
}
$env:RSI_MAX = $RsiMax
$env:MACD_ATR_K = $MacdAtrK
$env:SWEEP_WORKERS = $SweepWorkers
$env:SAVE_BT = $SaveBt

if ($RsiMin -ne "") {
    $env:RSI_MIN = $RsiMin
}
if ($GapEntryMax -ne "") {
    $env:GAP_ENTRY_MAX = $GapEntryMax
}
if ($ExitOnReverse -ne "") {
    $env:EXIT_ON_REVERSE = $ExitOnReverse
}
if ($UseIntradayResolution -ne "") {
    $env:USE_INTRADAY_RESOLUTION = $UseIntradayResolution
}
if ($IntradayInterval -ne "") {
    $env:INTRADAY_INTERVAL = $IntradayInterval
}
if ($IntradayTieBreak -ne "") {
    $env:INTRADAY_TIE_BREAK = $IntradayTieBreak
}

& $pythonExe (Join-Path $PSScriptRoot 'bt_sweep_inproc.py')

