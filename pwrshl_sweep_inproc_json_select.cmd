Get-Content sweep_inproc.json | ConvertFrom-Json |
    Sort-Object { $_.metrics.final_equity } -Descending |
    Select-Object -First 20 |
    ConvertTo-Json -Depth 6