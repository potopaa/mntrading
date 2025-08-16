#!/usr/bin/env bash
set -euo pipefail

UNIVERSE="${UNIVERSE:-BTC,ETH,SOL,BNB,XRP,ADA,MATIC,TRX,LTC,DOT}"
QUOTE="${QUOTE:-USDT}"
EXCHANGE="${EXCHANGE:-binance}"
SINCE_UTC="${SINCE_UTC:-}"
MIN_SAMPLES="${MIN_SAMPLES:-200}"
CORR_THRESHOLD="${CORR_THRESHOLD:-0.3}"
ALPHA="${ALPHA:-0.25}"
PAGE_SIZE="${PAGE_SIZE:-1000}"
TOPK_SCREEN="${TOPK_SCREEN:-50}"

NSPLITS="${NSPLITS:-3}"
GAP="${GAP:-5}"
MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-2000}"
ES="${ES:-50}"
PROBA_THR="${PROBA_THR:-0.5}"
FEE_RATE="${FEE_RATE:-0.0005}"

MIN_PROBA="${MIN_PROBA:-0.55}"
TOPK_TRADES="${TOPK_TRADES:-10}"
EQUITY="${EQUITY:-10000}"
LEVERAGE="${LEVERAGE:-1.0}"
LOOKBACK_BARS="${LOOKBACK_BARS:-2000}"

mkdir -p data/raw data/features/pairs data/datasets/pairs data/backtest_results data/models data/signals data/pairs data/portfolio

echo "[step] screen 1h"
ARGS=(./screen_pairs.py --universe "$UNIVERSE" --quote "$QUOTE" --source ccxt --exchange "$EXCHANGE" --min-samples "$MIN_SAMPLES" --corr-threshold "$CORR_THRESHOLD" --alpha "$ALPHA" --page-size "$PAGE_SIZE" --top-k "$TOPK_SCREEN")
if [ -n "$SINCE_UTC" ]; then ARGS+=(--since-utc "$SINCE_UTC"); fi
python "${ARGS[@]}"

SCREEN=$(ls -1t data/pairs/screened_pairs_*.json | head -n1)
echo "[info] using $SCREEN"

echo "[step] ingest 5m"
if [ -n "$SINCE_UTC" ]; then python ./main.py --mode ingest --symbols "$SCREEN" --timeframe 5m --limit 1000 --since-utc "$SINCE_UTC"; else python ./main.py --mode ingest --symbols "$SCREEN" --timeframe 5m --limit 1000; fi

echo "[step] features"
python ./main.py --mode features --symbols "$SCREEN" --beta-window 300 --z-window 300

echo "[step] dataset"
python ./main.py --mode dataset --pairs-manifest data/features/pairs/_manifest.json --label-type z_threshold --zscore-threshold 1.5 --lag-features 1

echo "[step] train"
python ./main.py --mode train --use-dataset --n-splits "$NSPLITS" --gap "$GAP" --max-train-size "$MAX_TRAIN_SIZE" --early-stopping-rounds "$ES" --proba-threshold "$PROBA_THR"

echo "[step] backtest"
python ./main.py --mode backtest --use-dataset --signals-from oof --proba-threshold "$PROBA_THR" --fee-rate "$FEE_RATE"

echo "[step] select"
python ./main.py --mode select --summary-path data/backtest_results/_summary.json --registry-out data/models/registry.json --sharpe-min 0.0 --maxdd-max 1.0 --top-k 20

echo "[step] promote"
python ./main.py --mode promote --production-map-out data/models/production_map.json

echo "[step] inference (production_map)"
python ./inference.py --registry data/models/production_map.json --pairs-manifest data/features/pairs/_manifest.json --timeframe 5m --limit 1000 --proba-threshold "$MIN_PROBA" --update --n-last 1 --out data/signals

echo "[step] aggregate"
python ./portfolio/aggregate_signals.py --signals-dir data/signals --pairs-manifest data/features/pairs/_manifest.json --min-proba "$MIN_PROBA" --top-k "$TOPK_TRADES" --scheme equal_weight --equity "$EQUITY" --leverage "$LEVERAGE"

echo "[step] report"
python ./portfolio/report_latest.py --orders-dir data/portfolio --backtests-dir data/backtest_results --lookback-bars "$LOOKBACK_BARS"

echo "[ok] bootstrap done"
