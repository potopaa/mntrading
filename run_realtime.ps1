param(
  [int]$IntervalSec = 300,
  [double]$MinProba = 0.55,
  [int]$TopK = 10,
  [double]$Equity = 10000,
  [double]$Leverage = 1.0
)

$ErrorActionPreference = "Stop"

function log($msg){ Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) }

while ($true) {
  try {
    log "inference (latest bar, update)"
    python inference.py `
      --registry data/models/production_map.json `
      --pairs-manifest data/features/pairs/_manifest.json `
      --signals-from auto `
      --proba-threshold $MinProba `
      --update --n-last 1 `
      --out data/signals

    log "aggregate -> orders"
    python portfolio/aggregate_signals.py `
      --signals-dir data/signals `
      --pairs-manifest data/features/pairs/_manifest.json `
      --min-proba $MinProba --top-k $TopK `
      --scheme equal_weight --equity $Equity --leverage $Leverage

    log "report"
    python portfolio/report_latest.py
  }
  catch {
    Write-Host "[err]" $_.Exception.Message
  }

  log ("sleep {0}s" -f $IntervalSec)
  Start-Sleep -Seconds $IntervalSec
}
