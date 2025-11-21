Get-Content runs.json | ConvertFrom-Json |
    Sort-Object { $_.metrics.final_equity } -Descending |
    Select-Object -First 5 |
    ConvertTo-Json -Depth 6
