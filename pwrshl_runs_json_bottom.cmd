Get-Content runs.json | ConvertFrom-Json |
    Sort-Object { $_.metrics.final_equity } -Ascending |
    Select-Object -First 20 |
    ConvertTo-Json -Depth 6
