# pattern 1
$env:GRID_VOL_SPIKE_M    = "1.2,1.6,2.0"
$env:GRID_MACD_ATR_K     = "0.10,0.15,0.20"
$env:GRID_TAKE_PROFIT_RR = "1.5,2.0"
$env:SAVE_BT = "1"
python bt_sweep.py > sweep_stage1.json

# pattern 2
$env:GRID_VOL_SPIKE_M    = "1.2,1.4,1.6"
$env:GRID_MACD_ATR_K     = "0.08:0.20:0.02"  # 0.08,0.10,...,0.20
$env:GRID_TAKE_PROFIT_RR = "1.2:2.8:0.2"
$env:GRID_GAP_ENTRY_MAX  = "0.03,0.05"
$env:GRID_RSI_MIN        = "42,45"
$env:GRID_RSI_MAX        = "68,70,72"
$env:SWEEP_TICKERS       = "3778.T,6702.T"
$env:SAVE_BT = "1"
python bt_sweep.py > sweep_multi.json

# pattern 3
$env:GRID_VOL_SPIKE_M    = "1.2,1.4,1.6"
$env:GRID_MACD_ATR_K     = "0.08:0.20:0.02"  # 0.08,0.10,...,0.20
$env:GRID_TAKE_PROFIT_RR = "1.2:2.8:0.2"
$env:GRID_GAP_ENTRY_MAX  = "0.03,0.05"
$env:GRID_RSI_MIN        = "42,45"
$env:GRID_RSI_MAX        = "68,70,72"
$env:SWEEP_TICKERS       = "3778.T,6702.T"
# $env:SAVE_BT = "1"
python bt_sweep_inproc.py > sweep_multi_inproc.json

# pattern 4
$env:SUPABASE_URL="https://frxhylrlaiaxlpkdsayg.supabase.co"
$env:SUPABASE_SERVICE_ROLE="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZyeGh5bHJsYWlheGxwa2RzYXlnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1Njk3NDY0OCwiZXhwIjoyMDcyNTUwNjQ4fQ.j6UJ3rFlLVDs0WeUg0DxYWIeHrkY2sjNixMAGFkbxkg"
$env:GRID_VOL_SPIKE_M    = "1.2,1.4,1.6"
$env:GRID_MACD_ATR_K     = "0.08:0.20:0.02"  # 0.08,0.10,...,0.20
$env:GRID_TAKE_PROFIT_RR = "1.2:2.8:0.2"
$env:GRID_GAP_ENTRY_MAX  = "0.03,0.05"
$env:GRID_RSI_MIN        = "42,45"
$env:GRID_RSI_MAX        = "68,70,72"
$env:SWEEP_TICKERS       = "3778.T,6702.T"
$env:SAVE_BT = "1"
python bt_sweep_inproc.py

# pattern 5
$env:SUPABASE_URL="https://frxhylrlaiaxlpkdsayg.supabase.co"
$env:SUPABASE_SERVICE_ROLE="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZyeGh5bHJsYWlheGxwa2RzYXlnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1Njk3NDY0OCwiZXhwIjoyMDcyNTUwNjQ4fQ.j6UJ3rFlLVDs0WeUg0DxYWIeHrkY2sjNixMAGFkbxkg"
$env:SWEEP_TICKERS       = "4506.T,9600.T"
$env:SAVE_BT = "1"
python bt_sweep_inproc.py