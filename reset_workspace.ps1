$ErrorActionPreference = "Stop"
$paths = @("data\raw","data\features","data\datasets","data\backtest_results","data\models","data\signals")
foreach ($p in $paths) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
$dirs = @("data\raw","data\features\pairs","data\datasets\pairs","data\backtest_results","data\models","data\signals","data\pairs")
$dirs | ForEach-Object { if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null } }
Write-Host "[ok] workspace reset."
