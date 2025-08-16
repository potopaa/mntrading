param(
  [int]$IntervalSec = 300,
  [double]$MinProba = 0.55,
  [int]$TopK = 10,
  [double]$Equity = 10000,
  [double]$Leverage = 1.0,
  [int]$LookbackBars = 2000
)

$ErrorActionPreference = "Stop"
function log($m){ Write-Host ("[rt] " + (Get-Date -Format "HH:mm:ss") + " " + $m) -ForegroundColor Yellow }

if (-not (Test-Path "data\features\pairs\_manifest.json")) { Write-Host "[err] run bootstrap first" -ForegroundColor Red; exit 1 }
if (-not (Test-Path "data\models\production_map.json")) { Write-Host "[err] run select/promote first" -ForegroundColor Red; exit 1 }

while ($true) {
  try {
    log "ingest 5m"
    python .\main.py --mode ingest --symbols data\features\pairs\_manifest.json --timeframe 5m --limit 1000

    log "inference"
    python .\inference.py --registry data\models\production_map.json --pairs-manifest data\features\pairs\_manifest.json --timeframe 5m --limit 1000 --proba-threshold $MinProba --update --n-last 1 --out data\signals

    log "aggregate -> orders"
    python .\portfolio\aggregate_signals.py --signals-dir data\signals --pairs-manifest data\features\pairs\_manifest.json --min-proba $MinProba --top-k $TopK --scheme equal_weight --equity $Equity --leverage $Leverage

    log "mini report"
    python .\portfolio\report_latest.py --orders-dir data\portfolio --backtests-dir data\backtest_results --lookback-bars $LookbackBars

    log "sleep $IntervalSec sec"
    Start-Sleep -Seconds $IntervalSec
  } catch {
    Write-Host "[rt][err] $($_.Exception.Message)" -ForegroundColor Red
    Start-Sleep -Seconds $IntervalSec
  }
}
