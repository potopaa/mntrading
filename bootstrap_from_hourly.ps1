param(
    # --- Screening (1h) ---
    [string]$Universe = "BTC,ETH,SOL,BNB,XRP,ADA,MATIC,TRX,LTC,DOT",
    [string]$Quote = "USDT",
    [string]$Exchange = "binance",
    [string]$SinceUTC = "",
    [int]$MinSamples = 200,
    [double]$CorrThreshold = 0.3,
    [double]$Alpha = 0.25,
    [int]$PageSize = 1000,
    [int]$TopKScreen = 50,

    # --- Train/Backtest ---
    [int]$NSplits = 3,
    [int]$Gap = 5,
    [int]$MaxTrainSize = 2000,
    [int]$ES = 50,
    [double]$ProbaThr = 0.5,
    [double]$FeeRate = 0.0005,

    # --- Portfolio aggregate/report ---
    [double]$MinProba = 0.55,
    [int]$TopKTrades = 10,
    [double]$Equity = 10000,
    [double]$Leverage = 1.0,
    [int]$LookbackBars = 2000
)

$ErrorActionPreference = "Stop"
function log($msg) { Write-Host ("[step] " + $msg) -ForegroundColor Cyan }

# 0) Dirs
$dirs = @("data/raw","data/features/pairs","data/datasets/pairs","data/backtest_results","data/models","data/signals","data/pairs","data/portfolio")
$dirs | ForEach-Object { if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null } }

# 1) Hourly screening
log "screen 1h (universe=$Universe, since_utc=$SinceUTC)"
$screenArgs = @(
  ".\screen_pairs.py","--universe",$Universe,"--quote",$Quote,"--source","ccxt","--exchange",$Exchange,
  "--min-samples",$MinSamples,"--corr-threshold",$CorrThreshold,"--alpha",$Alpha,"--page-size",$PageSize,"--top-k",$TopKScreen
)
if ($SinceUTC) { $screenArgs += @("--since-utc",$SinceUTC) }
python @screenArgs

$screen = (Get-ChildItem -Path "data\pairs\screened_pairs_*.json" | Sort-Object LastWriteTime -Desc | Select-Object -First 1).FullName
if (-not $screen) { throw "No screened_pairs_*.json produced" }
Write-Host "[info] using pairs: $screen"

# 2) Ingest 5m
log "ingest 5m"
if ($SinceUTC) {
  python .\main.py --mode ingest --symbols $screen --timeframe 5m --limit 1000 --since-utc $SinceUTC
} else {
  python .\main.py --mode ingest --symbols $screen --timeframe 5m --limit 1000
}
if (-not (Test-Path "data\raw\ohlcv.parquet")) { throw "missing data\raw\ohlcv.parquet" }

# 3) Features
log "features (pairs)"
python .\main.py --mode features --symbols $screen --beta-window 300 --z-window 300
if (-not (Test-Path "data\features\pairs\_manifest.json")) { throw "missing data\features\pairs\_manifest.json" }

# 4) Dataset
log "dataset (X,y)"
python .\main.py --mode dataset --pairs-manifest data\features\pairs\_manifest.json --label-type z_threshold --zscore-threshold 1.5 --lag-features 1
if (-not (Test-Path "data\datasets\_manifest.json")) { throw "missing data\datasets\_manifest.json" }

# 5) Train
log "train (NSplits=$NSplits, Gap=$Gap, MaxTrainSize=$MaxTrainSize, ES=$ES)"
python .\main.py --mode train --use-dataset --n-splits $NSplits --gap $Gap --max-train-size $MaxTrainSize --early-stopping-rounds $ES --proba-threshold $ProbaThr
if (-not (Test-Path "data\models\_train_report.json")) { Write-Warning "train report not found" }

# 6) Backtest
log "backtest (signals-from=oof)"
python .\main.py --mode backtest --use-dataset --signals-from oof --proba-threshold $ProbaThr --fee-rate $FeeRate
if (-not (Test-Path "data\backtest_results\_summary.json")) { throw "missing backtest summary" }

# 7) Select
log "select -> registry.json"
python .\main.py --mode select --summary-path data\backtest_results\_summary.json --registry-out data\models\registry.json --sharpe-min 0.0 --maxdd-max 1.0 --top-k 20
if (-not (Test-Path "data\models\registry.json")) { throw "missing registry" }

# 8) Promote
log "promote -> production_map.json"
python .\main.py --mode promote --production-map-out data\models\production_map.json
if (-not (Test-Path "data\models\production_map.json")) { throw "missing production map" }

# 9) Inference (ВАЖНО: используем production_map.json)
log "inference (latest bar, update=on)"
python .\inference.py --registry data\models\production_map.json --pairs-manifest data\features\pairs\_manifest.json --timeframe 5m --limit 1000 --proba-threshold $MinProba --update --n-last 1 --out data\signals

# 10) Aggregate → orders
log "aggregate signals -> orders"
python .\portfolio\aggregate_signals.py --signals-dir data\signals --pairs-manifest data\features\pairs\_manifest.json --min-proba $MinProba --top-k $TopKTrades --scheme equal_weight --equity $Equity --leverage $Leverage

# 11) Mini report
log "mini report"
python .\portfolio\report_latest.py --orders-dir data\portfolio --backtests-dir data\backtest_results --lookback-bars $LookbackBars

Write-Host "`n[ok] pipeline finished."
