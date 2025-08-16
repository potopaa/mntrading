$ErrorActionPreference = "Stop"

$rm = @(
  "data\raw","data\features","data\datasets","data\backtest_results",
  "data\models","data\signals","data\pairs","mlruns"
)
foreach ($p in $rm) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }

$mk = @(
  "data\raw","data\features\pairs","data\datasets\pairs",
  "data\backtest_results","data\models\pairs","data\signals","data\pairs"
)
foreach ($d in $mk) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }

Write-Host "[ok] workspace reset."
