#!/usr/bin/env bash
set -euo pipefail
INTERVAL_SEC="${INTERVAL_SEC:-300}"
MIN_PROBA="${MIN_PROBA:-0.55}"
TOPK="${TOPK:-10}"
EQUITY="${EQUITY:-10000}"
LEVERAGE="${LEVERAGE:-1.0}"
LOOKBACK_BARS="${LOOKBACK_BARS:-2000}"

[ -f data/features/pairs/_manifest.json ] || { echo "[err] run bootstrap first"; exit 1; }
[ -f data/models/production_map.json ] || { echo "[err] run select/promote first"; exit 1; }

while true; do
  date +"[rt] %F %T ingest"
  python ./main.py --mode ingest --symbols data/features/pairs/_manifest.json --timeframe 5m --limit 1000

  date +"[rt] %F %T inference (production_map)"
  python ./inference.py --registry data/models/production_map.json --pairs-manifest data/features/pairs/_manifest.json --timeframe 5m --limit 1000 --proba-threshold "$MIN_PROBA" --update --n-last 1 --out data/signals

  date +"[rt] %F %T aggregate"
  python ./portfolio/aggregate_signals.py --signals-dir data/signals --pairs-manifest data/features/pairs/_manifest.json --min-proba "$MIN_PROBA" --top-k "$TOPK" --scheme equal_weight --equity "$EQUITY" --leverage "$LEVERAGE"

  date +"[rt] %F %T report"
  python ./portfolio/report_latest.py --orders-dir data/portfolio --backtests-dir data/backtest_results --lookback-bars "$LOOKBACK_BARS"
  echo "[rt] sleep ${INTERVAL_SEC}s"; sleep "$INTERVAL_SEC"
done
