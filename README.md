# mntrading — Market-Neutral Pairs Trading (API + UI + Docker)

Automated market-neutral **pairs trading** on Binance (5-minute bars) with a reproducible, MLOps-friendly pipeline and a small web UI.

**Pipeline (current state, Aug 2025)**  
`screen(1h) → ingest(5m) → features(pairs) → dataset(X,y) → train(CV) → backtest → select → promote → inference → aggregate → mini-report`

Key capabilities
- **Time-series CV** with **embargo (`gap`)** and optional **sliding window (`max_train_size`)**.
- Models: **RandomForest**, **XGBoost**, **LightGBM**; per-fold **early stopping** for XGB/LGBM and optional **probability calibration** for RF.
- Validation-time **trading metrics** on folds (Sharpe / MaxDD) to choose the champion.
- Robust dataset builder: **duplicate-timestamp dedup**, **lag features**, **forecast `horizon`** (shift labels into the future) and safe index alignment.
- **FastAPI** service with orchestration endpoints + **Streamlit** UI (auto-refresh retains state).
- **MLflow** experiment tracking; artifacts per pair; **registry** and **production map** for inference.
- **Docker Compose** setup (API + UI) with a shared `./data` volume.

> ⚠️ Research code. Not investment advice.

---

## Project layout

```
api/server.py                   # FastAPI (run/pipeline orchestration + artifacts)
ui/streamlit_app.py             # Streamlit dashboard (Operations, Reports, Artifacts)
main.py                         # CLI entry point for stages (ingest/features/dataset/train/…)
screen_pairs.py                 # Screen pairs on hourly data
features/spread.py              # Pair features: OLS beta/alpha, spread, z-score
features/labels.py              # Build supervised datasets (X,y); dedup index; horizon/lag support
models/train.py                 # Walk-forward CV, OOF, validation trading metrics, MLflow
backtest/backtest.py            # Backtests using OOF / model / z-rule signals
inference.py                    # Online/batch inference → JSONL signals
portfolio/aggregate_signals.py  # Aggregate latest signals → orders (JSON/CSV)
portfolio/report_latest.py      # Mini report (JSON + charts)
Dockerfile, docker-compose.yml, requirements.txt
```

Outputs (created on first run)
```
data/raw/ohlcv.parquet, ohlcv_meta.json
data/pairs/screened_pairs_*.json
data/features/pairs/*.parquet, _manifest.json
data/datasets/pairs/*.parquet, _manifest.json
data/models/_train_report.json, registry.json, production_map.json, per-pair artifacts
data/backtest_results/_summary.json, per-pair backtest files
data/signals/*.jsonl
data/portfolio/orders_*.{json,csv}, report_*/summary.json, report_*/charts/*.png
```

---

## Quickstart — local

### 1) Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> Windows notes: we force UTF-8 in API subprocesses and replaced fancy symbols with ASCII to avoid encoding glitches in logs.

### 2) Start services (local)
Terminal A — **API**:
```bash
uvicorn api.server:app --reload --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
```
Terminal B — **UI**:
```bash
python -m streamlit run ui/streamlit_app.py
# By default a random port is chosen. To fix the port:
# python -m streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

### 3) Full bootstrap from the UI
Open the Streamlit app → **🚀 Operations** → **“Run bootstrap_quick”**.  
This runs:
1h screen → 5m ingest → features → dataset → train → backtest → select → promote → inference → aggregate → mini-report.

The **Last task logs** panel auto-refreshes (checkbox persists between refreshes).

### 4) Short cycle (ingest → inference → aggregate → report)
In the UI use **Run short cycle**, or via API:
```bash
# default params
curl -fsS "http://127.0.0.1:8000/run/short_cycle_now"

# custom params
curl -fsS -X POST "http://127.0.0.1:8000/run/short_cycle"   -H "Content-Type: application/json"   -d '{"timeframe":"5m","limit":1000,"proba_threshold":0.55,
       "top_k":10,"equity":10000,"leverage":1.0,"lookback_bars":2000}'
```

---

## CLI — stage by stage (optional)

```bash
# 1) Screen (1h) → candidates
python screen_pairs.py   --universe "BTC,ETH,SOL,BNB,XRP,ADA,MATIC,TRX,LTC,DOT" --quote USDT   --source ccxt --exchange binance --since-utc 2025-01-01   --min-samples 200 --corr-threshold 0.3 --alpha 0.25 --page-size 1000 --top-k 50

# 2) Ingest (5m) for selected pairs
python main.py --mode ingest   --symbols data/pairs/screened_pairs_*.json   --timeframe 5m --limit 1000 --since-utc 2025-01-01

# 3) Features (pairs)
python main.py --mode features   --symbols data/pairs/screened_pairs_*.json   --beta-window 300 --z-window 300

# 4) Dataset (X,y) — robust build
python main.py --mode dataset   --pairs-manifest data/features/pairs/_manifest.json   --label-type z_threshold --zscore-threshold 1.5   --lag-features 1 --horizon 0

# 5) Train — time-series CV (walk-forward)
python main.py --mode train --use-dataset   --n-splits 3 --gap 5 --max-train-size 2000   --early-stopping-rounds 50 --proba-threshold 0.5

# 6) Backtest (OOF/model/z)
python main.py --mode backtest --use-dataset   --signals-from oof --proba-threshold 0.5 --fee-rate 0.0005

# 7) Select champions → registry.json
python main.py --mode select   --summary-path data/backtest_results/_summary.json   --registry-out data/models/registry.json   --sharpe-min 0.0 --maxdd-max 1.0 --top-k 20

# 8) Promote → production_map.json
python main.py --mode promote   --production-map-out data/models/production_map.json

# 9) Inference (latest signals)
python inference.py   --registry data/models/production_map.json   --pairs-manifest data/features/pairs/_manifest.json   --timeframe 5m --limit 1000   --proba-threshold 0.55 --update --n-last 1   --out data/signals

# 10) Aggregate → orders + Mini report
python portfolio/aggregate_signals.py   --signals-dir data/signals   --pairs-manifest data/features/pairs/_manifest.json   --min-proba 0.55 --top-k 10   --scheme equal_weight --equity 10000 --leverage 1.0

python portfolio/report_latest.py   --orders-dir data/portfolio   --backtests-dir data/backtest_results   --lookback-bars 2000
```

---

## REST API (FastAPI)

Docs: `GET /docs` (Swagger) and `GET /openapi.json`.

**Run endpoints**
- `POST /run/bootstrap_quick` — full pipeline (screen → … → report).
- `POST /run/short_cycle` — ingest + inference + aggregate + report (with JSON body).
- `GET  /run/short_cycle_get` — same via query params.
- `GET  /run/short_cycle_now` — same with defaults.
- `GET  /tasks/last` — last task status + captured logs.

**Artifacts and latest data**
- `GET /artifacts/production_map`
- `GET /artifacts/backtest_summary`
- `GET /artifacts/train_report`
- `GET /artifacts/features_manifest`
- `GET /artifacts/datasets_manifest`
- `GET /signals/latest`
- `GET /orders/latest`
- `GET /report/latest`
- `GET /health`

> The Streamlit UI reads `MNTRADING_API` (default `http://127.0.0.1:8000`; in Docker it’s `http://api:8000`).

---

## Streamlit UI

- **Operations**: buttons to trigger pipelines, with a **Last task logs** panel.  
  Auto-refresh uses a stateful rerun (not meta-refresh), so the checkbox **stays on**.
- **Reports**: latest signals and orders; mini report (Sharpe/MaxDD/CumRet + charts).  
  When the portfolio contains a single pair, the portfolio chart equals the pair chart; duplicates are hidden.
- **Artifacts**: view production map, train report, features/datasets manifests.
- Download buttons for generated `orders_*.json` and `orders_*.csv` (from `data/portfolio/`).

Env var:
```bash
# local
export MNTRADING_API=http://127.0.0.1:8000
# docker (already set in compose)
MNTRADING_API=http://api:8000
```

---

## Docker Compose

Two services: **api** (Uvicorn) and **ui** (Streamlit), sharing `./data`.

```bash
docker compose build
docker compose up -d
# API → http://localhost:8000  |  UI → http://localhost:8501
```

- Stable ports: 8000 (API) and 8501 (UI).
- Data persistence: the host `./data` is mounted as `/app/data` in both containers.
- Health checks ensure UI waits for API.
- Rebuild after code changes: `docker compose up -d --build`.

> Optional: a scheduler sidecar can periodically call `/run/short_cycle_now`. See **Roadmap** below.

---

## Modeling details

- Features per pair: `pa, pb, beta, alpha, spread, z` (windows configurable).  
- Labels: `z_threshold` with configurable `zscore_threshold`.  
- `lag_features` applies a shift to features; `horizon` (or `forecast_horizon`) shifts labels to predict `H` bars ahead (no look-ahead bias).  
- CV: walk-forward splits with `gap` (embargo) and optional `max_train_size` (sliding window).  
- Best model chosen by validation trading metrics (Sharpe/MaxDD) + AUC (where applicable).  
- MLflow logs runs; we store `oof_predictions`, configs, fold metrics, and selected artifacts per pair.

---

## Troubleshooting

- **422 Validation Error** (API): send JSON body for `POST` routes, or use `GET /run/short_cycle_get` with query parameters.
- **Streamlit port error (WinError 10013)**: run with explicit `--server.address 127.0.0.1 --server.port 8501` and allow in the firewall.
- **Unicode / mojibake in logs**: we force `PYTHONIOENCODING=utf-8`; project no longer prints fancy arrows; tqdm glyphs are cosmetic.
- **“cannot reindex on an axis with duplicate labels”**: fixed — dataset builder deduplicates timestamps and aligns indices safely.
- **XGBoost model load warning**: harmless; models are loaded via MLflow’s sklearn wrapper.
- **No charts / identical charts**: when only one pair is active, portfolio == that pair; duplicates are hidden in the UI.

---

## Roadmap (what’s next)

- 🔁 **Scheduler sidecar** in `docker-compose` (e.g. curl every 5–10 minutes → `/run/short_cycle_now`).  
- 📈 **Monitoring & alerts** (health, timeouts, error push to Telegram/Slack, log rotation).  
- 🧠 **Model tuning** (horizon/lag sweeps, feature importance, blending, improved selection).  
- 📦 **MLflow remote** (S3/MinIO) + tags/notes; update around MLflow stages deprecation.  
- 🧪 **Tests & CI** (unit/integration, pre-commit, GitHub Actions).  
- 💸 **Execution layer** (paper/live trading, risk limits, slippage).

---

## License / disclaimer

MIT-style (unless stated otherwise). For research purposes only.
Not financial advice.
