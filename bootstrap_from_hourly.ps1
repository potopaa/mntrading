param(
  [string]$SinceUtc = "2025-01-01T00:00:00Z",
  [string]$Syms = "BTC/USDT,ETH/USDT,SOL/USDT",
  [int]$LimitH1 = 5000,
  [int]$LimitM5 = 50000
)

$ErrorActionPreference = "Stop"

function log($msg){ Write-Host ("`n== {0} ==" -f $msg) }

log "Ingest H1 (for screening)"
python main.py --mode ingest --symbols $Syms --timeframe 1h --since-utc $SinceUtc --limit $LimitH1

log "Screen pairs (optional)"
python screen_pairs.py --raw-parquet data/raw/ohlcv.parquet --min-bars 2000 --min-corr 0.6 --top-k 200

log "Ingest M5 (for training)"
python main.py --mode ingest --symbols $Syms --timeframe 5m --since-utc $SinceUtc --limit $LimitM5

log "Features"
$PairsJson = (Get-ChildItem -Path data/pairs -Filter "screened_pairs_*.json" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
if (-not $PairsJson) { $PairsJson = "" }
if ($PairsJson) {
  python main.py --mode features --symbols $PairsJson
} else {
  python main.py --mode features --symbols $Syms
}

log "Train (multi-model)"
python main.py --mode train --use-dataset --n-splits 3 --gap 5 --max-train-size 2000 --early-stopping-rounds 50

log "Backtest / Select / Promote"
python main.py --mode backtest
python main.py --mode select --top-k 20
python main.py --mode promote

Write-Host "`n[ok] bootstrap complete."
